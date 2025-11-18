# -*- coding: utf-8 -*-
import os
import re
import warnings
from functools import cached_property, partial
from typing import Literal, Optional, get_args

import numpy as np
import pandas as pd
import pymupdf
from scipy.ndimage import find_objects, label

from ._constants import QUALI_DRIVERS
from .drivers import Drivers
from .models.classification import SessionEntryImport, SessionEntryObject
from .models.driver import (
    DriverImport,
    DriverObject,
    RoundEntryImport,
    RoundEntryObject,
    TeamDriverImport,
    TeamDriverObject,
)
from .models.foreign_key import (
    PitStopForeignKeys,
    RoundEntryForeignKeys,
    SessionEntryForeignKeys,
    TeamDriverForeignKeys,
)
from .models.lap import LapImport, LapObject
from .models.pit_stop import PitStopData, PitStopObject
from .utils import (
    DPI,
    Page,
    ParsingError,
    TextBlock,
    duration_to_millisecond,
    quali_lap_times_to_json,
    time_to_timedelta,
)

pd.set_option('future.no_silent_downcasting', True)

PracticeSessionT = Literal['fp', 'fp1', 'fp2', 'fp3']
RaceSessionT = Literal['race', 'sprint']
QualiSessionT = Literal['quali', 'sprint_quali']

DRIVERS = Drivers()

WHITE_STRIP_MIN_HEIGHT = 10  # A table should end with a white strip with at least 10px height
LINE_MIN_VGAP = 5  # If two horizontal lines are vertically separated by less than 5px, they are
                   # considered to be the same line


class BaseParser:
    """Base class for all parsers

    Provides some common functionality, such as locating the title, parsing tables by grid lines,
    etc. Not meant to be instantiated directly, but rather to be inherited by others.
    """
    @staticmethod
    def _parse_table_by_grid(
            page: Page,
            vlines: list[float],
            hlines: list[float],
            tol: float = 2,
            header_included: bool = True
    ) -> pd.DataFrame:
        """Parse the table cell by cell, defined by lines separating the columns and rows

        :param vlines: x-coords. of vertical lines separating the cols. Table left and right
                       boundaries need to be included
        :param hlines: y-coords. of horizontal lines separating the rows. Table top and bottom
                       boundaries need to be included
        :param tol: tolerance for text and cell positioning. In principle, all texts should fall
                    inside the cell's bounding box. Default is 2 pixels, i.e. if text is within 2px
                    of the cell's boundary, it is considered to be inside the cell. See #33
        :param header_included: whether the first row is header/col. names. Default is True
        """
        cells = []
        for i in range(len(hlines) - 1):
            row = []
            for j in range(len(vlines) - 1):
                text = ''
                l, t, r, b = vlines[j], hlines[i], vlines[j + 1], hlines[i + 1]

                # Get text inside the cell, defined by the vertical and horizontal lines
                # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html
                """
                For each cell defined by the `vlines` and `hlines`, we get text inside it. However,
                texts that are partially inside the cell will also be captured by pymupdf. So we
                need to check whether the found text is totally or partially inside the cell's
                bounding box, and do not want to false include other texts. However, we do want to
                allow for a bit of tolerance, as `hlines` or `vlines` are not always perfectly
                positioned. The tolerance is set to 2 pixels in general. But for PDFs that have
                smaller page margin, i.e. text font size or line height are bigger in these PDFs,
                we need to increase the tolerance. See #33.
                """
                cell = page.get_text('blocks', clip=(l, t, r, b), small_area=True)
                if cell:
                    # Usually, one cell is one line of text. The only exception is Andrea Kimi
                    # Antonelli. His name is too long and thus (maybe) wrapped into two lines (#42)
                    if len(cell) > 1:
                        if (len(cell) == 2  # noqa: PLR2004
                                and ('antonelli' in cell[0][4].lower()
                                     or 'antonelli' in cell[1][4].lower())):
                            text = cell[0][4].strip() + ' ' + cell[1][4].strip()
                        else:
                            raise ValueError(
                                f'Expected one text block in row {i}, col. {j} on p.{page.number} '
                                f'in {page.file}. Found {len(cell)}: {cell}'
                            )
                    elif len(cell) == 1:
                        cell = cell[0]
                        if cell[4].strip():
                            bbox = cell[0:4]
                            if bbox[0] < l - tol or bbox[2] > r + tol \
                                    or bbox[1] < t - tol or bbox[3] > b + tol:
                                raise ValueError(
                                    f"Found text outside the cell's area in row {i}, col. {j} on "
                                    f"p.{page.number} in {page.file}: {cell}"
                                )
                            text = cell[4].strip()
                    else:
                        raise ValueError(
                            f'Expected one text block in row {i}, col. {j} on p.{page.number} in '
                            f'{page.file}. Found {len(cell)}: {cell}'
                        )
                row.append(text)
            cells.append(row)

        if header_included:
            return pd.DataFrame(cells[1:], columns=cells[0])
        else:
            return pd.DataFrame(cells)

    @staticmethod
    def _detect_cols(
            page: Page,
            clip: tuple[float, float, float, float],
            col_min_gap: float = 1.1,
            min_black_line_length: float = 0.9
    ) -> list[TextBlock]:
        """
        Search for table header/cols. in the `clip` area. Returns a list of (l, t, r, b, col. name)

        The high-level idea is:

        1. convert the `clip` area into an image
        2. for each character, mask it into black rectangle using `scipy.ndimage.label`. This is
           possible because all black pixels are text, all texts are in black, and everything else
           is white
        3. merge multiple black rectangles into a single one if they are sufficiently close to
           each other
        4. run OCR on each black rectangle to get the text inside it. And the four corners of the
           rectangle are the bounding box of the text

        :param col_min_gap: Minimum gap between two cols., relative to the width of the median
                            thinnest char. Default is 1.1, i.e. 10% tolerance. If two chars. are
                            horizontally closer than this gap, they are considered to be in the
                            same word. We many have extremely thin char., like a dot. So we can't
                            really base this on such extreme value. From thin to fat, we may have
                            ".", "I", "a", "W". Usually the width of "a" is a good reference. So we
                            use the width of the median thinnest char. here
        :param min_black_line_length: Minimum length of a black line, relative to the width of
                                      the `clip` area. Default is 0.9, i.e. 90% of the width of
                                      the `clip` area. Any black horizontal line longer than this
                                      is ignored
        """
        # Get the pixmap of `clip` area
        pixmap = page.get_pixmap(clip=clip, dpi=DPI)
        arr = np.ndarray([pixmap.h, pixmap.w, 3], dtype=np.uint8, buffer=pixmap.samples)
        del pixmap

        # Drop rows and cols. with almost all black pixels (those are lines not text). After this
        # step, all black pixels should be texts only
        """
        As there are usually only three colours in the PDF: black (RGB ~= 21), white (0), and grey
        (184 or 232), use RGB = 128 as a cutoff for "black"
        """
        arr = arr[np.mean(arr < 128, axis=(1, 2)) < min_black_line_length]  # noqa: PLR2004
        arr = arr[:, np.mean(arr < 128, axis=(0, 2)) < min_black_line_length]  # noqa: PLR2004

        # Mask text with rectangles, i.e. label connected components
        labelled, n_features = label(arr < 128)  # noqa: PLR2004
        if n_features == 0:
            return []
        slices = sorted(find_objects(labelled), key=lambda x: x[1].start)

        # Visualise the detected slices
        # for s in slices:
        #     arr[s[0], s[1]] = [0, 0, 0]

        # Merge the rectangles/slices that are close to each other
        """
        "Close" is defined as smaller than the width of the median thinnest char., which is often
        "a". The gap between two cols. are usually very wide, except "NAT" and "ENTRANT", between
        which the gap is roughly a normal whitespace. If we pick a too wide "close", "NAT" and
        "ENTRANT" will be recognised as a single text block "NAT ENTRANT". If "close" is too
        narrow, then "NAT" may be broken into "N", "A", and "T". Generally speaking, using the
        median thinnest char.'s width (w/ ~10% buffer) satisfies both conditions.
        """
        med_slice_width = np.median([s[1].stop - s[1].start for s in slices]) * col_min_gap
        merged_slices = []
        for s in slices:
            if not merged_slices:
                merged_slices.append(s)
            elif s[1].start - merged_slices[-1][1].stop < med_slice_width:
                merged_slices[-1] = (
                    slice(min(merged_slices[-1][0].start, s[0].start),
                          max(merged_slices[-1][0].stop, s[0].stop)),
                    slice(min(merged_slices[-1][1].start, s[1].start),
                          max(merged_slices[-1][1].stop, s[1].stop)),
                    merged_slices[-1][2]
                )
            else:
                merged_slices.append(s)
        del slices

        # OCR the merged slices
        cols = []
        for s in merged_slices:
            l, t, r, b = page._transform_bbox(
                original_bbox=clip,
                new_bbox=(0, 0, arr.shape[1], arr.shape[0]),
                bbox=(s[1].start, s[0].start, s[1].stop, s[0].stop)
            )
            text = page.get_text('text', clip=(l - 1, t - 1, r + 1, b + 1)).strip()
            if text:
                # The "/" in "KM/H" is often mis-OCRed. Manually correct it here
                if re.match(r'KM\SH', text):
                    text = 'KM/H'
                cols.append(TextBlock(text=text, bbox=(l, t, r, b)))
            else:
                raise ParsingError(f'Unable to OCR text in the table header inside '
                                   f'({l:.2f}, {t:.2f}, {r:.2f}, {b:.2f}) on p.{page.number} in '
                                   f'{page.file}')
        return cols

    @staticmethod
    def _detect_rows(
            page: Page,
            clip: tuple[float, float, float, float],
            min_black_line_length: float = 0.9
    ) -> list[TextBlock]:
        """
        Search for table rows in the `clip` area. Returns a list of TextBlock of the row names

        See `_detect_cols()` for detals

        :param min_black_line_length: Minimum length of a black vertical line, relative to the
                                      height of the `clip` area. Default is 0.9, i.e. 90% of the
                                      height of the `clip` area. Any black vertical line longer
                                      than this is ignored
        """
        # Get the pixmap of `clip` area
        pixmap = page.get_pixmap(clip=clip, dpi=DPI)
        arr = np.ndarray([pixmap.h, pixmap.w, 3], dtype=np.uint8, buffer=pixmap.samples)
        del pixmap

        # Drop cols. with almost all black pixels (those are lines not text). After this step, all
        # black pixels should be texts only
        """
        As there are usually only three colours in the PDF: black (RGB ~= 21), white (0), and grey
        (184 or 232), use RGB = 128 as a cutoff for "black"
        """
        arr = arr[np.mean(arr < 128, axis=(1, 2)) < min_black_line_length]  # noqa: PLR2004

        # Mask text with rectangles, i.e. label connected components
        labelled, n_features = label(arr < 128)  # noqa: PLR2004
        if n_features == 0:
            return []
        slices = sorted(find_objects(labelled), key=lambda x: x[0].start)

        # Merge the rectangles/slices that are close to each other
        min_slice_height = min([s[0].stop - s[0].start for s in slices])
        merged_slices = []
        for s in slices:
            if not merged_slices:
                merged_slices.append(s)
            elif s[0].start - merged_slices[-1][0].stop < min_slice_height:
                merged_slices[-1] = (
                    slice(min(merged_slices[-1][0].start, s[0].start),
                          max(merged_slices[-1][0].stop, s[0].stop)),
                    slice(min(merged_slices[-1][1].start, s[1].start),
                          max(merged_slices[-1][1].stop, s[1].stop)),
                    merged_slices[-1][2]
                )
            else:
                merged_slices.append(s)
        del slices

        # Get text/OCR the merged slices
        rows = []
        for s in merged_slices:
            l, t, r, b = page._transform_bbox(
                original_bbox=clip,
                new_bbox=(0, 0, arr.shape[1], arr.shape[0]),
                bbox=(s[1].start, s[0].start, s[1].stop, s[0].stop)
            )
            text = page.get_text('text', clip=(l - 1, t - 1, r + 1, b + 1)).strip()
            if text:
                rows.append(TextBlock(text=text, bbox=(l, t, r, b)))
            else:
                raise ParsingError(f'Unable to get the text inside the row '
                                   f'({l:.2f}, {t:.2f}, {r:.2f}, {b:.2f}) on p.{page.number} in '
                                   f'{page.file}')
        return rows


class EntryListParser:
    def __init__(
            self,
            file: str | os.PathLike,
            year: int,
            round_no: int
    ):
        self.file = file
        self.year = year
        self.round_no = round_no
        self.df = self._parse()

    def _parse_table_by_grid(
            self,
            page: pymupdf.Page,
            vlines: list[float],
            hlines: list[float],
            line_height: float,
            tol: float = 2
    ) -> pd.DataFrame:
        """Parse the table cell by cell, defined by lines separating the columns and rows

        This overrides the method from `BaseParser` to handle the superscripts. The superscript
        indicates which reserve driver is driving whose car. E.g., Antonelli (reserve) is driving
        Hamilton's (normal) car, then driver No. 44 and driver No. 12 have the same superscript.

        :param page: the page to parse
        :param vlines: x-coords. of vertical lines separating the cols. Table left and right
                       boundaries need to be included
        :param hlines: y-coords. of horizontal lines separating the rows. Table top and bottom
                       boundaries need to be included
        :param line_height: height of a usual row, i.e. the height of a car No. text. Will use this
                            to detect the end of the main driver table and the start of the reserve
                            driver table
        :param tol: tolerance for bbox. of text. Default is 2 pixels. See #33
        """
        cells = []
        for i in range(len(hlines) - 1):
            row = []
            superscripts = []
            for j in range(len(vlines) - 1):

                # Check if there is an unusually big gap between two consecutive horizontal lines.
                # If so, then we are now at the gap between the main table and the reserve driver
                # table, so skip the current row (, which is the gap)
                if hlines[i + 1] - hlines[i] > line_height * 1.5:
                    break

                # Get text(s) in the cell
                l, t, r, b = vlines[j], hlines[i], vlines[j + 1], hlines[i + 1]
                cell = page.get_text('dict', clip=(l, t, r, b))
                spans = []
                for block in cell['blocks']:
                    for line in block['lines']:
                        for span in line['spans']:
                            if span['text'].strip():
                                bbox = span['bbox']
                                # Need to check if the found text is indeed in the cell's bbox.
                                # PyMuPDF is notoriously bad for not respecting `clip` parameter.
                                # We give `tol` pixels tolerance. See #33
                                if bbox[0] >= l - tol and bbox[2] <= r + tol \
                                        and bbox[1] >= t - tol and bbox[3] <= b + tol:
                                    spans.append(span)

                # Check if any superscript
                # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-analyze-font-characteristics  # noqa: E501
                # for font flags
                superscript = None
                if len(spans) == 1:    # Only one text, which is the usual text, and no superscript
                    row.append(spans[0]['text'].strip())
                elif len(spans) == 2:  # Two texts. Should be one usual text and one superscript  # noqa: E501, PLR2004
                    n_superscripts = 0
                    n_regular_text = 0
                    for span in spans:
                        if span['flags'] == 0:
                            regular_text = span['text'].strip()
                            n_regular_text += 1
                        elif span['flags'] == 1:
                            superscript = span['text'].strip()
                            n_superscripts += 1
                        else:
                            raise ValueError(
                                f'Unable to infer whether the text is a regular text or a '
                                f'superscript in cell in row {i}, col. {j} on p.{page.number} in '
                                f'{self.file}: {cell}'
                            )
                    # If we found two regular texts above, then have to decide which is the
                    # superscript using font size (#48). In principle superscript font size should
                    # be sufficiently smaller than the regular text. We use 20% diff. as a cutoff
                    if n_superscripts == 2 or n_regular_text == 2:  # noqa: PLR2004
                        temp = (spans[0]['size'] + spans[1]['size']) / 2
                        if (spans[0]['size'] - spans[1]['size']) / temp > 0.2:  # noqa: PLR2004
                            superscript = spans[1]['text'].strip()
                            regular_text = spans[0]['text'].strip()
                        elif (spans[1]['size'] - spans[0]['size']) / temp > 0.2:  # noqa: PLR2004
                            superscript = spans[0]['text'].strip()
                            regular_text = spans[1]['text'].strip()
                        else:
                            raise ValueError(f'Cannot determine which text is superscript in '
                                             f'row {i}, col {j} on p.{page.number} in '
                                             f'{self.file}: {cell}')
                    row.append(regular_text)
                    superscripts.append(superscript)
                else:
                    raise ValueError(f'Found more than two text blocks in the cell in row {i}, '
                                     f'col. {j} on p.{page.number} in {self.file}. Expected only '
                                     f'one or two texts. Found: {cell}')

            # Only the zero-th cell in each row (the car No. col.) can have a superscript, so after
            # processing all cells in the row, should get at most one single superscript per row
            if len(superscripts) > 1:
                raise ValueError(f'Found multiple superscripts in row {i} on p.{page.number} in '
                                 f'{self.file}. Expected one or no superscript per row. Found: '
                                 f'{superscripts}')
            if row:  # Empty iff. we are at the gap between the main and reserve driver table
                row.append(superscripts[0] if superscripts else None)  # Add superscript to the row
                cells.append(row)

        # Convert to df.
        df = pd.DataFrame(cells)
        if df.shape[1] == 5:    # noqa: PLR2004
            df.columns = ['car_no', 'driver', 'nat', 'team', 'constructor']
        elif df.shape[1] == 6:  # noqa: PLR2004
            df.columns = ['car_no', 'driver', 'nat', 'team', 'constructor', 'reserve']
        else:
            raise ValueError(f'Expected either 5 or 6 cols. in {self.file}. Found: '
                             f'{df.columns.tolist()}')
        df.car_no = df.car_no.astype(int)
        assert df.car_no.is_unique, f'Car No. is not unique in {self.file}: {df.car_no.tolist()}'

        # Clean up the reserve driver relationship
        if 'reserve' not in df.columns:
            df['reserve'] = False
            return df
        df['reserve_for'] = None
        for i in df.reserve.dropna().unique():  # For each superscript
            temp = df[df.reserve == i]          # Find the driver(s) with this superscript
            # Should have two drivers: one main and one reserve for the main
            if len(temp) == 2:  # noqa: PLR2004
                assert temp.car_no.nunique() == 2, (  # noqa: PLR2004
                    f'Expected two different drivers for superscript {i} in {self.file}. Found '
                    f'{temp.car_no.nunique()}'
                )
                df.loc[df[df.reserve == i].index[1], 'reserve_for'] = temp.car_no.iloc[0]
            # Handle the case where there is only one single driver with this superscript. This
            # means a driver is incorrectly indicated as having a reserve driver, e.g. copy-paste
            # error in the raw PDF in 2024 Chinese (#23)
            # TODO: this may create problems. E.g., the parser misses a superscript and then misses
            #       a reserve driver
            elif len(temp) == 1:
                df.loc[df.reserve == i, 'reserve'] = None
                warnings.warn(
                    f'Driver {temp.driver.iloc[0]} is indicated as being or having a reserve '
                    f'driver but no associated driver was found in {self.file}'
                )
            else:
                raise ValueError(
                    f'Expected exactly two drivers with superscript {i} in {self.file}. Found '
                    f'{temp.driver.tolist()}'
                )
        df.reserve = df['reserve_for'].notna()
        return df

    def _parse(self) -> pd.DataFrame:
        """
        :return: Df. with cols. of ["car_no", "driver", "nat", "team", "constructor"]
        """
        # Go to the page with "No.", "Driver", "Nat", "Team", and "Constructor"
        doc = pymupdf.open(self.file)
        found = False
        for page in doc:
            text = page.get_text('text')
            if 'No.' in text and 'Driver' in text and 'Nat' in text and 'Team' in text and \
                    'Constructor' in text:
                found = True
                break
        if not found:
            raise ValueError(f'Could not find any page containing entry list table in {self.file}')

        # The top y-coord. of the table is below "Driver"
        """
        1. locate the positions of "Driver"'s. One page may have multiple "Driver"'s, e.g. 2024
           Mexican. We only pick the topmost one
        2. make sure the found "Driver" is indeed the table header. That is, in the same height,
           we should find "No.", "Nat", "Team", and "Constructor" as well
        """
        driver = page.search_for('Driver')
        driver.sort(key=lambda x: x.y0)
        driver = driver[0]
        for col in ['No.', 'Nat', 'Team', 'Constructor']:
            temp = page.search_for(col, clip=(0, driver.y0, page.bound()[2], driver.y1))
            assert len(temp) == 1, f'Cannot locate "Driver" in {self.file}'

        # Table headers
        headers = {}
        for col in ['No.', 'Driver', 'Nat', 'Team', 'Constructor']:
            temp = page.search_for(col, clip=(0, driver.y0, page.bound()[2], driver.y1))
            assert len(temp) == 1, f'Expected one "{col}", got {len(temp)} in {self.file}'
            headers[col.lower().strip('.')] = temp[0]

        # The leftmost x-coord. of the table is the leftof "No."
        l = headers['no'].x0

        # Right is simply the page's right boundary
        r = page.bound()[2]

        # Bottom of the table
        """
        It's slightly more difficult to determine the bottom of the table, as there is no line
        delineating the bottom of the table. Below the table, we have stewards' names, so we can't
        use the bottom of the page as the bottom of the table either. So instead, we search for
        digits below "No.". The last car No.'s position is the the bottom of the table.

        When we say "the table", there can actually be two tables: the usual table for drivers, and
        the reserve driver table. Regardless of the existence of the reserve driver table, the
        above method always identifies the bottom correctly. `self._parse_table_by_grid()` will
        handle the reserve driver table properly by looking at the superscripts.
        """
        car_nos = page.get_text(
            'words',
            clip=(headers['no'].x0, headers['no'].y1, headers['no'].x1, page.bound()[3])
        )
        car_nos = [i for i in car_nos if i[4].isdigit()]
        for i in car_nos:
            assert np.isclose(i[0], l, atol=1), \
                f'Car No. {i[4]} is not vertically aligned with "No." in {self.file}'
            assert i[2] < driver.x0, \
                f'Car No. {i[4]} is not to the left of "Driver" in {self.file}'

        # Lines separating the columns
        aux_vlines = [
            l,
            headers['driver'].x0,
            headers['nat'].x0,
            headers['team'].x0,
            headers['constructor'].x0,
            r
        ]

        # Line gap and height
        """
        Line gap defined as the gap between the bottom of a car No. and the top of the next car No.
        If we have ten drivers, then there are nine line gaps. We take the second largest of these
        gaps as a usual line gap. There is an usually big gap between the main table and the
        reserve driver table, which is the largest gap and we want to skip it, so use the second
        largest.
        """
        line_gap = sorted([abs(car_nos[i + 1][1] - car_nos[i][3])
                           for i in range(len(car_nos) - 1)])[-2]
        """
        We use 1/3 of the line height as a sanity check, here and throughout this repo. 1/3
        basically means, if `line_gap` is 10px, then anything taller than 3.3px is fine. This seems
        crazy but actually works quite well. Here are the two reasons (not only matters there but
        everywhere in the parser):

        1. false negative: will a text area be mistakenly excluded because it's short? Unlikely,
           because if the area has any text, that text definitely needs to occupy 1/3 of a normal
           line height. Think about the height of "o", "p", and "l": "o" is shorter than "l" but
           still needs to occupy a significant portion of the line height
        2. false negative: will a non-text area be mistakenly included? No, because text is always
           in black, and sometimes has grey background. We always have such colour checks when
           dealing with empty areas, so they won't be mistakenly included. Actually the opposite.
           This is the reason why we use a very conservative 1/3 here, because some empty areas are
           too short, so need to be very conservative when detecting these non-text empty areas.
           E.g., the empty area between the main table and fastest lap in 2024 Spanish vs the same
           empty area in 2025 Austrian
        """
        line_height = np.mean([i[3] - i[1] for i in car_nos])
        assert line_gap < line_height / 3, (
            f'Line gap {line_gap} is too big relative to line height {line_height} in {self.file}'
        )

        # Lines separating the rows
        """
        These row separators are usually the midpoints of a car No. and the next car No. However,
        if we are at the last row of the main table, or the first row of the reserve driver table,
        such midpoints are not row separators, but rather the midpoint of bottom of the main table
        and the top of the reserve driver table. So we need to handle these cases separately, using
        `line_gap` above.
        """
        # Zero-th row separator is between "No." and the top of the first car No.
        aux_hlines = [(headers['no'].y1 + car_nos[0][1]) / 2]
        # The rest are midpoints or determined by the line height and gap as specified above
        for i, _ in enumerate(car_nos[:-1]):
            curr_line_gap = car_nos[i + 1][1] - car_nos[i][3]
            # If the current row and the next row are sufficiently close, then they belong to the
            # same table, so use the midpoint
            if curr_line_gap < line_gap * 2:  # Zero-th row is always fine as above
                aux_hlines.append((car_nos[i][3] + car_nos[i + 1][1]) / 2)
            # If they are vertically too far from each other, then the current row is the last row
            # of the main table, so use the bottom of the current row + line gap. Also
            # need to append the top of the next row, which is the first row of the reserve driver
            # table
            else:
                aux_hlines.append(car_nos[i][3] + line_gap)
                aux_hlines.append(car_nos[i + 1][1] - line_gap)
        # The last row's bottom is the bottom of the last row + line gap
        aux_hlines.append(car_nos[-1][3] + line_gap)

        # Get the table
        tol = 2 if l > 40 else 3  # noqa: PLR2004
        df = self._parse_table_by_grid(page, aux_vlines, aux_hlines, line_height, tol)

        def to_json() -> list[dict]:
            """
            Create RoundEntry for each driver in the entry list, and create DriverObject for
            drivers that are not yet in Jolpica
            """
            drivers: list[dict] = []
            new_driver_objects: list[DriverObject] = []
            new_team_drivers: list[dict] = []
            for x in df.itertuples():
                # Check if the driver exists in Jolpica. If not, create a DriverObject and
                # TeamDriverObject (mark him as a junior driver) for him
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter('always')
                    driver_id = DRIVERS.get(year=self.year, full_name=x.driver)
                    for warn in w:
                        if 'Creating a new driver ID' in str(warn.message):
                            new_driver_objects.append(
                                DriverObject(
                                    reference=driver_id,
                                    forename=' '.join(x.driver.split(' ')[:-1]),
                                    surname=x.driver.split(' ')[-1],
                                    country_code=x.nat
                                )
                            )
                            new_team_drivers.append(
                                TeamDriverImport(
                                    object_type='TeamDriver',
                                    foreign_keys=TeamDriverForeignKeys(
                                        year=self.year,
                                        team_reference=x.constructor,
                                        driver_reference=driver_id
                                    ),
                                    objects=[
                                        TeamDriverObject(
                                            role=2
                                        )
                                    ]
                                ).model_dump(exclude_unset=True)
                            )

                drivers.append(RoundEntryImport(
                    object_type='RoundEntry',
                    foreign_keys=RoundEntryForeignKeys(
                        year=self.year,
                        round=self.round_no,
                        team_reference=x.constructor,
                        driver_reference=driver_id
                    ),
                    objects=[
                        RoundEntryObject(
                            car_number=x.car_no
                        )
                    ]
                ).model_dump(exclude_unset=True))

            if new_driver_objects:
                warnings.warn('New drivers found in entry list PDF')
                drivers.append(DriverImport(objects=new_driver_objects)
                               .model_dump(exclude_none=True))  # Need default foreign key here so
                drivers.extend(new_team_drivers)                # can't `exclude_unset`
            return drivers

        df.to_json = to_json
        return df


class PracticeParser(BaseParser):
    def __init__(
            self,
            classification_file: str | os.PathLike,
            lap_times_file: Optional[str | os.PathLike],
            year: int,
            round_no: int,
            session: PracticeSessionT
    ):
        self.classification_file = classification_file
        self.lap_times_file = lap_times_file
        self.session = 'fp1' if session == 'fp' else session  # Sprint weekend FP renamed to FP1
        self.year = year
        self.round_no = round_no
        self._check_session()

    def _check_session(self) -> None:
        if self.session not in get_args(PracticeSessionT):
            raise ValueError(f'Invalid session: {self.session}. '
                             f'Valid sessions are: {get_args(PracticeSessionT)}')
        return

    @cached_property
    def is_pdf_complete(self) -> bool:
        return self.lap_times_file is not None

    @cached_property
    def classification_df(self) -> pd.DataFrame:
        return self._parse_classification()

    @cached_property
    def lap_times_df(self) -> Optional[pd.DataFrame]:
        if self.lap_times_file:
            return self._parse_lap_times()
        else:
            warnings.warn('Lap times PDF is missing. Can get fastest laps only from the '
                          'classification PDF')
            return self._apply_fallback_fastest_laps()

    def _parse_classification(self) -> pd.DataFrame:
        """Parse "(First/Second/Third) Practice Final Classification" PDF

        The output dataframe has columns [driver No., laps completed, total time,
        finishing position, finishing status, fastest lap time, fastest lap speed, fastest lap No.]
        """
        # Find the page with "Practice Session Classification", on which the table is located
        doc = pymupdf.open(self.classification_file)
        classification = []
        for i in range(len(doc)):
            page = Page(doc[i], file=self.classification_file)
            classification = page.search_for('Practice Session Classification')
            if classification:
                break
        if not classification:
            doc.close()
            raise ValueError(f'"Practice Session Classification" not found on any page in '
                             f'{self.classification_file}')

        # Position of "Practice Session Classification", below which is the table
        b_classification = classification[0].y1

        # Find the first black horizontal line below "Practice Session Classification"
        if black_line := page.search_for_black_line(clip=(0, b_classification, page.w, page.h)):
            t_table_body = sorted(black_line)[0]  # Topmost black line below "Final Classification"
        else:
            raise ParsingError(f'Cannot find the black line separating table header and table '
                               f'body below "Practice Session Classification" on p.{page.number} '
                               f'in {self.classification_file}')

        # Get text/line height
        temp = page.get_text('blocks', clip=(0, b_classification, page.w, t_table_body))
        if not temp:
            raise ParsingError(f'Cound not find any text in the table header on p.{page.number} '
                               f'in {self.classification_file}')
        line_height = np.mean([i[3] - i[1] for i in temp])

        # The first white strip below the table header, which is the bottom of the table
        white_strip = page.search_for_white_strip(clip=(0, t_table_body, page.w, page.h),
                                                  height=line_height / 3)
        if white_strip:
            b_table = sorted(white_strip)[0]
        else:
            raise ParsingError(f'Could not find table bottom by white strip on p.{page.number} in '
                               f'{self.classification_file}')

        # Get table col. names and their positions
        cols = self._detect_cols(page,
                                 clip=(0, b_classification + 1, page.w, t_table_body - 1),
                                 col_min_gap=1)
        if not cols:
            raise ParsingError(f'Could not detect cols. in the table header on p.{page.number} in '
                               f'{self.classification_file}')
        cols = {i.text: i for i in cols}
        if set(cols.keys()) != {'NO', 'DRIVER', 'NAT', 'ENTRANT', 'TIME', 'LAPS', 'GAP', 'INT',
                                'KM/H', 'TIME OF DAY'}:
            raise ParsingError(f'Got unexpected or miss some table cols. on p.{page.number} in '
                               f'{self.classification_file}: {cols}')

        # Vertical lines separating the columns
        vlines = [
            0,
            cols['NO'].l,
            (cols['NO'].r + cols['DRIVER'].l) / 2,
            cols['NAT'].l - 1,
            (cols['NAT'].r + cols['ENTRANT'].l) / 2,
            1.5 * cols['TIME'].l - 0.5 * cols['TIME'].r,  # Left of "TIME" - half-width of "TIME"
            cols['LAPS'].l,
            cols['LAPS'].r,
            (cols['GAP'].r + cols['INT'].l) / 2,
            (cols['INT'].r + cols['KM/H'].l) / 2,
            cols['TIME OF DAY'].l,
            cols['TIME OF DAY'].r
        ]

        # Horizontal lines separating the rows
        hlines = page.search_for_grey_white_rows(clip=(0, t_table_body + 1, page.w, b_table + 1),
                                                 min_height=line_height / 3)

        # Parse the table using the grid above
        df = self._parse_table_by_grid(
            page=page,
            vlines=vlines,
            hlines=hlines,
            header_included=False
        )
        assert df.shape[1] == 11, (  # noqa: PLR2004
            f'Expected 11 cols. on p.{page.number} in {self.classification_file}. Got '
            f'{df.columns.tolist()}'
        )
        df.columns = ['position'] + list(cols.keys())
        df = df.replace('', None)  # Empty strings --> None, so `duration_to_millisecond` can skip
                                   # e.g. empty "TIME" cells without error
        # Set col. names
        del df['NAT']
        df = df.rename(columns={
            'position': 'finishing_position',
            'NO': 'car_no',
            'DRIVER': 'driver',
            'ENTRANT': 'team',
            'TIME': 'fastest_lap_time',
            'LAPS': 'laps_completed',
            'GAP': 'gap',
            'INT': 'int',
            'KM/H': 'avg_speed',
            'TIME OF DAY': 'fastest_lap_calender_time'
        })
        df.finishing_position = df.finishing_position.astype(int)
        df.car_no = df.car_no.astype(int)
        df.laps_completed = df.laps_completed.astype(int)
        df.fastest_lap_time = df.fastest_lap_time.apply(duration_to_millisecond)

        """
        We don't really know the finishing status of the driver from the table. E.g., in 2024
        Australian FP1, Albon crashed, but before that, he already set several valid laps, so his
        name is in the PDF, without any mark about the crash. So we set the finishing status to be
        missing for everyone, because we can't infer that from the PDF.

        And all drivers in the table will be classified, because as long as they make a lap that
        counts, they are classified and in the PDF.
        """

        def to_json() -> list[dict]:
            return df.apply(
                lambda x: SessionEntryImport(
                    object_type="SessionEntry",
                    foreign_keys=SessionEntryForeignKeys(
                        year=self.year,
                        round=self.round_no,
                        session=self.session.upper(),
                        car_number=x.car_no
                    ),
                    objects=[
                        SessionEntryObject(
                            position=x.finishing_position,
                            is_classified=True,
                            status=None,
                            laps_completed=x.laps_completed,
                            fastest_lap_rank=x.finishing_position  # It's FP so finishing position
                        )                                          # is the fastest lap ranking
                    ]
                ).model_dump(exclude_none=True, exclude_unset=True),
                axis=1
            ).tolist()

        df.to_json = to_json
        return df

    def _parse_lap_times(self) -> pd.DataFrame:
        """Parse "Practice Session Lap Times" PDF"""
        # TODO: this is identical to race lap analysis and quali. lap times parser. Consider
        #       refactor/unify them
        doc = pymupdf.open(self.lap_times_file)
        dfs = []
        for page in doc:
            # Find "Lap Times"
            page = Page(page, file=self.lap_times_file)  # noqa: PLW2901
            quali_lap_times = page.search_for('Lap Times')
            if len(quali_lap_times) != 1:
                raise ParsingError(f'Find none or multiple "Lap Times" on p.{page.number} in '
                                   f'{self.lap_times_file}')
            b_lap_times = quali_lap_times[0].y1

            # Find the white strip immediately below "Lap Times", below which are the tables
            white_strip = page.search_for_white_strip(clip=(0, b_lap_times, page.w, page.h))
            if not white_strip:
                raise ParsingError(f'Expect at least a white strip below "Lap Times" on '
                                   f'p.{page.number} in {self.lap_times_file}. Found: '
                                   f'{white_strip}')
            t_all_drivers = white_strip[0]

            # Find all black horizontal lines (see RaceParser._parse_lap_analysis for details)
            black_lines = page.search_for_black_line(clip=(0, t_all_drivers, page.w, page.h),
                                                     min_length=0.25)
            if not black_lines:
                raise ParsingError(f'Could not find any black line below "Lap Times" on '
                                   f'p.{page.number} in {self.lap_times_file}')

            # Each line should be the separator between a table header and its body
            t_table_headers = []
            t_drivers = []
            b_tables = []
            for i in range(len(black_lines) - 1, -1, -1):
                # Table header is vertically between the black line and the white strip immediately
                # above the black line
                black_line = black_lines[i]
                white_strip = page.search_for_white_strip(clip=(0, 0, page.w, black_line))
                if not white_strip:
                    raise ParsingError(f'Could not find any white strips above the black line '
                                       f'at {black_line} on p.{page.number} in '
                                       f'{self.lap_times_file}. Found: {white_strip}')
                t_table_header = sorted(white_strip)[-1]
                header = page.get_text(clip=(0, t_table_header, page.w, black_line))

                # If no table header found, then it's the last black line at the bottom of the
                # page. Drop it
                if not ('NO' in header and 'TIME' in header):
                    black_lines.pop(i)
                    continue
                t_table_headers.insert(0, t_table_header)

                # The driver name is above the table header and the next white strip above
                if len(white_strip) < 2:  # noqa: PLR2004
                    raise ParsingError(f'Expect at least two white strips above the black line at '
                                       f'{black_line} on p.{page.number} in '
                                       f'{self.lap_times_file}. Found: {white_strip}')
                t_drivers.insert(0, white_strip[-2])

                # Table bottom is the next white strip below the black line
                white_strip = page.search_for_white_strip(clip=(0, black_line, page.w, page.h))
                if not white_strip:
                    raise ParsingError(f'Could not find any white strip below the black line at '
                                       f'{black_line} on p.{page.number} in {self.lap_times_file}')
                b_tables.insert(0, white_strip[0])

            # Two tables are vertically separated by a vertical white strip
            page.set_rotation(90)
            table_separators = page.search_for_white_strip(
                clip=(page.h - b_tables[-1], 0, page.h - t_drivers[0], page.w),
                height=0.03 * page.w  # White strip should occupy at least 3% of the page width
            )
            page.set_rotation(0)
            # A driver has at least one table, so at least two white strips: one to the left of the
            # table and the other to the right of it
            if len(table_separators) < 2:  # noqa: PLR2004
                raise ParsingError(f'Expect at least two vertical white strips below driver names '
                                   f'on p.{page.number} in {self.lap_times_file}. Found: '
                                   f'{table_separators}')

            # Loop through each table
            for i in range(len(t_drivers)):
                t_driver = t_drivers[i] + 1
                t_table_header = t_table_headers[i] + 1
                b_table_header = black_lines[i] - 1
                b_table = b_tables[i] + 1
                for j in range(0, len(table_separators) - 1):
                    l_table = max(0, table_separators[j] - 1)
                    r_table = table_separators[j + 1] + 1

                    # Get the driver name and car No.
                    driver = page.get_text(clip=(l_table, t_driver, r_table, t_table_header))
                    if not driver.strip():  # E.g., four tables on a page. The second row only has
                        continue            # one table, so we will have missing's here
                    car_no = re.match(r'^(\d+)\s+\D+$', driver.strip())
                    if car_no:
                        car_no = int(car_no.group(1))
                    else:
                        raise ParsingError(f'Could not parse car No. in '
                                           f'({l_table:.2f}, {t_driver:.2f}, {r_table:.2f}, '
                                           f'{t_table_header:.2f}) on p.{page.number} in '
                                           f'{self.lap_times_file}: {driver}')

                    # Find the vertical white strip separating the two tables for the driver
                    page.set_rotation(90)
                    white_strip = page.search_for_white_strip(
                        clip=(page.h - b_table, l_table, page.h - t_table_header, r_table),
                        height=1  # Any height is fine
                    )
                    page.set_rotation(0)
                    # Table left, separator, and right. Three in total
                    if len(white_strip) != 3:  # noqa: PLR2004
                        raise ParsingError(f'Expected exactly three vertical white strips in '
                                           f'({l_table:.2f}, {t_table_header:.2f}, '
                                           f'{r_table:.2f}, {b_table:.2f}) on p.'
                                           f'{page.number} in {self.lap_times_file}. '
                                           f'Found: {white_strip}')
                    m_table = white_strip[1] + 1

                    # Parse each of the two tables of the driver
                    for l_tab, r_tab in [(l_table, m_table), (m_table, r_table)]:
                        # Refine table bottom
                        white_strip = page.search_for_white_strip(
                            clip=(l_tab, b_table_header, r_tab, page.h)
                        )
                        if not white_strip:
                            raise ParsingError(f'Could not find any white strip below the table in '
                                               f'({l_tab:.2f}, {b_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table:.2f}) on p.'
                                               f'{page.number} in {self.lap_times_file}')
                        b_table = white_strip[0]

                        # Get table header
                        cols = self._detect_cols(
                            page,
                            clip=(l_tab, t_table_header, r_tab, b_table_header),
                            col_min_gap=3,
                            min_black_line_length=0.5
                        )
                        if len(cols) != 2:  # noqa: PLR2004
                            raise ParsingError(f'Expected exactly two cols. in '
                                               f'({l_tab:.2f}, {t_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table_header:.2f}) on p.'
                                               f'{page.number} in {self.lap_times_file}. '
                                               f'Found: {cols}')
                        if cols[0].text != 'NO' or cols[1].text != 'TIME':
                            raise ParsingError(f'Expected "LAP" and "TIME" to be the two col. '
                                               f'names in ({l_tab:.2f}, {t_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table_header:.2f}) on p.'
                                               f'{page.number} in {self.lap_times_file}. '
                                               f'Found: {cols}')
                        l_tab = cols[0].l - 1  # More accurate table left boundary  # noqa: PLW2901

                        # Vertical lines separating the two cols.
                        vlines = [l_tab, (cols[0].r + cols[1].l) / 2, r_tab]

                        # Horizontal lines are located by the grey and white rows
                        hlines = page.search_for_grey_white_rows(
                            clip=(l_tab, b_table_header - 1, r_tab, b_table + 1),
                            min_height=np.mean([i.b - i.t for i in cols]) - 1,
                            min_width=0.5
                        )

                        # Parse the table
                        df = self._parse_table_by_grid(page=page,
                                                       vlines=vlines,
                                                       hlines=hlines,
                                                       header_included=False)
                        if df.empty:  # E.g., DNS so no lap at all
                            continue
                        df.columns = [i.text.lower() for i in cols]
                        df = df.rename(columns={'no': 'lap_no', 'time': 'lap_time'})
                        df['lap_time_deleted'] = False

                        # Check if any crossed-out lap times
                        for k in range(len(hlines) - 1):
                            clip = (vlines[1], hlines[k], vlines[2], hlines[k + 1])
                            if page.has_horizontal_line(clip):
                                df.loc[k, 'lap_time_deleted'] = True

                        # Indicator for pit stop
                        df['pit'] = df.lap_no.str.contains('P', regex=False)
                        df.lap_no = df.lap_no.str.rstrip(' P').astype(int)
                        df['car_no'] = int(car_no)
                        dfs.append(df)

        # Clean up
        df = pd.concat(dfs, ignore_index=True)
        df.lap_no = df.lap_no.astype(int)
        df = df[df.lap_no != 1]  # First lap's lap time is calendar time of the lap, not lap time
        df.car_no = df.car_no.astype(int)
        df = df.replace('', None)  # So empty cell will become NaN when cast to float
        df.lap_time = df.lap_time.apply(duration_to_millisecond)

        # Check if fastest lap here is the same as that in classification PDF
        classification_df = self.classification_df[['car_no', 'fastest_lap_time']]
        temp = (df[(df.lap_time_deleted == False) & (df.pit == False)]  # noqa: E712
                .assign(lap_time_milliseconds=lambda x: x.lap_time.str['milliseconds'])
                .sort_values(['car_no', 'lap_time_milliseconds'], ascending=True)
                .groupby('car_no', as_index=False)
                .first()[['car_no', 'lap_no', 'lap_time']]
                .merge(classification_df[classification_df.fastest_lap_time.notna()],
                       on='car_no',  # The above `.notna()` excludes drivers without any valid lap
                       how='outer',  # time, e.g. 2024 Azerbaijan FP1 Ocon
                       validate='1:1',
                       indicator=True))
        # TODO: what if a driver has two laps with identical fastest lap time? Label the first one
        #       as the fastest lap and second as not?
        if (temp._merge != 'both').any():
            raise AssertionError(f'Found some cars only appearing in one but not both of the lap '
                                 f'times and classification PDFs:\n'
                                 f'{temp[temp._merge != "both"].to_string(index=False)}')
        if (temp.lap_time != temp.fastest_lap_time).any():
            diff = temp[temp.lap_time != temp.fastest_lap_time]
            raise AssertionError(f'Fastest lap time in lap times PDF does not match the one in '
                                 f'classification PDF\n:{diff.to_string(index=False)}')

        # Create the indicator for fastest lap
        df = df.merge(temp[['car_no', 'lap_no']],
                      on=['car_no', 'lap_no'],
                      how='left',
                      validate='m:1',
                      indicator=True)
        df['is_fastest_lap'] = (df._merge == 'both')
        del df['_merge']

        def to_json() -> list[dict]:
            temp = df.copy()
            temp['lap'] = temp.apply(
                lambda x: LapObject(
                    number=x.lap_no,
                    time=x.lap_time,
                    is_deleted=x.lap_time_deleted,
                    is_entry_fastest_lap=x.is_fastest_lap
                ),
                axis=1
            )
            temp = temp.groupby('car_no')[['lap']].agg(list).reset_index()
            temp['session_entry'] = temp.car_no.map(
                lambda x: SessionEntryForeignKeys(
                    year=self.year,
                    round=self.round_no,
                    session=self.session,
                    car_number=x
                )
            )
            return temp.apply(
                lambda x: LapImport(
                    object_type='Lap',
                    foreign_keys=x.session_entry,
                    objects=x.lap
                ).model_dump(exclude_unset=True),
                axis=1
            ).tolist()

        df.to_json = to_json  # TODO: bad practice
        return df

    def _apply_fallback_fastest_laps(self) -> pd.DataFrame:
        """
        Get fastest laps (and no other lap at all) from classification PDF, in case lap times PDF
        is not available.
        """
        fastest_laps = self.classification_df[['car_no', 'fastest_lap_time']].dropna(how='any')

        def to_json() -> list[dict]:
            return fastest_laps.apply(
                lambda x: LapImport(
                    object_type='Lap',
                    foreign_keys=SessionEntryForeignKeys(
                        year=self.year,
                        round=self.round_no,
                        session=self.session,
                        car_number=x.car_no
                    ),
                    objects=[
                        LapObject(
                            number=None,  # can't know the lap No. from classification PDF
                            time=x.fastest_lap_time,
                            is_deleted=False,
                            is_entry_fastest_lap=True
                        )
                    ]
                ).model_dump(exclude_unset=True),
                axis=1
            ).tolist()

        fastest_laps.to_json = to_json  # TODO: bad practice
        return fastest_laps


class RaceParser(BaseParser):
    def __init__(
            self,
            classification_file: str | os.PathLike,
            lap_analysis_file: str | os.PathLike,
            history_chart_file: str | os.PathLike,
            lap_chart_file: str | os.PathLike,
            year: int,
            round_no: int,
            session: RaceSessionT
    ):
        self.classification_file = classification_file
        self.lap_analysis_file = lap_analysis_file
        self.history_chart_file = history_chart_file
        self.lap_chart_file = lap_chart_file
        self.session = session
        self.year = year
        self.round_no = round_no
        self._check_session()
        # self._cross_validate()

    @cached_property
    def is_pdf_complete(self) -> bool:
        """Check if we have all lap times PDFs. If not, won't be able to get lap times df"""
        if self.classification_file is None:
            raise FileNotFoundError("Classification PDF is missing. Can't parse anything")
        if (self.lap_analysis_file is None or self.history_chart_file is None
                or self.lap_chart_file is None):
            return False
        return True

    @cached_property
    def classification_df(self) -> pd.DataFrame:
        return self._parse_classification()

    @cached_property
    def starting_grid(self) -> pd.DataFrame:
        """
        A bit confusing here. We get the starting grid from lap chart PDF. And we need the same PDF
        for lap times. So need to parse lap chart PDF twice, which is slow. To save time, we only
        parse it once in `self._parse_lap_times()`, and in this method we save the starting grid
        to the attribute `self.starting_grid`. There is something wrong/redundant here. Not fixed

        TODO: refactor
        """
        if not self.is_pdf_complete:
            raise FileNotFoundError("Lap chart, history chart, or lap time PDFs is missing. Can't "
                                    "parse starting grid or lap times")
        _ = self.lap_times_df
        return self.starting_grid

    @cached_property
    def lap_times_df(self) -> pd.DataFrame:
        if not self.is_pdf_complete:
            raise FileNotFoundError("Lap chart, history chart, or lap time PDFs is missing. Can't "
                                    "parse starting grid or lap times")
        self.starting_grid = None
        return self._parse_lap_times()

    def _check_session(self) -> None:
        """Check that the input session is valid. Raise an error otherwise"""
        if self.session not in get_args(RaceSessionT):
            raise ValueError(f'Invalid session: {self.session}. '
                             f'Valid sessions are: {get_args(RaceSessionT)}')
        return

    def _parse_classification(self) -> pd.DataFrame:
        """Parse "Race/Sprint Race Final Classification" PDF

        The output dataframe has columns [driver No., laps completed, total time,
        finishing position, finishing status, fastest lap time, fastest lap speed, fastest lap No.]
        """
        # Find the page with "Final Classification", on which the table is located
        doc = pymupdf.open(self.classification_file)
        classification = []
        for i in range(len(doc)):
            page = Page(doc[i], file=self.classification_file)
            if '.pdf' in page.get_text():  # Fix #59
                continue
            classification = page.search_for('Final Classification')
            if classification:
                break
            classification = page.search_for('Provisional Classification')
            if classification:
                warnings.warn('Found and using provisional classification, not the final one')
                break
        if not classification:
            doc.close()
            raise ParsingError(f'"Final Classification" or "Provisional Classification" not found '
                               f'on any page in {self.classification_file}')

        # Bottom position of "Final Classification", below which is the table header/col. names
        b_classification = classification[0].y1

        # Locate the first long black horizontal line below "Final Classification". This is the
        # line separating the table header and table body
        black_line = page.search_for_black_line(
            clip=(0, b_classification, page.bound()[2], page.bound()[3])
        )
        if black_line:
            t_table_body = sorted(black_line)[0]  # Topmost black line below "Final Classification"
        else:
            raise ParsingError(f'Cannot find the black line separating table header and table '
                               f'body below "Final Classification" on p.{page.number} in '
                               f'{self.classification_file}')

        # Page width. This is the right boundary of the table
        w = page.bound()[2]

        # Get text height. Need this to understand line height, vertical gaps between rows, etc.
        temp = page.get_text('blocks', clip=(0, b_classification, w, t_table_body))
        if not temp:
            raise ParsingError(f'Cound not find any text in the table header on p.{page.number} '
                               f'in {self.classification_file}')
        line_height = np.mean([i[3] - i[1] for i in temp])

        # Find the first white strip below the table header. This is the bottom of the table
        white_strip = page.search_for_white_strip(clip=(0, t_table_body, w, page.bound()[3]),
                                                  height=line_height / 3)
        if white_strip:
            b_table = sorted(white_strip)[0]
        else:
            raise ParsingError(f'Could not find table bottom by white strip on p.{page.number} in '
                               f'{self.classification_file}')

        # Get table col. names and their positions
        cols = self._detect_cols(page,
                                 clip=(0, b_classification + 1, w, t_table_body - 1),
                                 col_min_gap=1)
        if not cols:
            raise ParsingError(f'Could not detect cols. in the table header on p.{page.number} in '
                               f'{self.classification_file}')
        cols = {i.text: i for i in cols}
        if set(cols.keys()) != {'NO', 'DRIVER', 'NAT', 'ENTRANT', 'LAPS', 'TIME', 'GAP', 'INT',
                                'KM/H', 'FASTEST', 'ON', 'PTS'}:
            raise ParsingError(f'Got unexpected or miss some table cols. on p.{page.number} in '
                               f'{self.classification_file}: {cols}')

        # Vertical lines separating the cols.
        vlines = [
            0,
            cols['NO'].l,
            (cols['NO'].r + cols['DRIVER'].l) / 2,
            cols['NAT'].l - 1,
            (cols['NAT'].r + cols['ENTRANT'].l) / 2,
            cols['LAPS'].l,
            cols['LAPS'].r ,
            (cols['TIME'].r + cols['GAP'].l) / 2,
            (cols['GAP'].r + cols['INT'].l) / 2,
            (cols['INT'].r + cols['KM/H'].l) / 2,
            cols['FASTEST'].l - 1,
            cols['FASTEST'].r + 1,
            cols['PTS'].l,
            cols['PTS'].r
        ]

        # Horizontal lines separating the rows
        hlines = page.search_for_grey_white_rows(clip=(0, t_table_body + 1, w, b_table + 1),
                                                 min_height=line_height / 3)

        # Parse the table using the grid above
        df = self._parse_table_by_grid(
            page=page,
            vlines=vlines,
            hlines=hlines,
            header_included=False
        )
        assert df.shape[1] == 13, \
            f'Expected 13 cols, got {df.shape[1]} in {self.classification_file}'  # noqa: PLR2004
        df.columns = ['position'] + [i for i in cols.keys()]
        # TODO: very bad. This assumes dict. is ordered

        # Check if there is a "NOT CLASSIFIED" table below the main table
        not_classified = page.search_for('NOT CLASSIFIED')

        # If yes, repeat the above for the "NOT CLASSIFIED" table
        if not_classified:
            t_table_body = not_classified[0].y1
            white_strip = page.search_for_white_strip(clip=(0, t_table_body, w, page.bound()[3]),
                                                      height=line_height / 3)
            if white_strip:
                b_table = sorted(white_strip)[0]
            else:
                raise ParsingError(
                    f'Could not find the bottom of "NOT CLASSIFIED" table by white strip on '
                    f'p.{page.number} in {self.classification_file}'
                )
            hlines = page.search_for_grey_white_rows(clip=(0, t_table_body, w, b_table),
                                                     min_height=line_height / 3)
            not_classified = self._parse_table_by_grid(
                page=page,
                vlines=vlines,
                hlines=hlines,
                header_included=False
            )
            not_classified = not_classified.sort_index().reset_index(drop=True)
            assert not_classified.shape[1] == 13, \
                (f'Expected 13 columns for "NOT CLASSIFIED" table , got '  # noqa: PLR2004
                 f'{not_classified.shape[1]} in {self.classification_file}')
            not_classified.columns = df.columns
            not_classified.position = None  # No finishing position for unclassified drivers

        else:
            # No unclassified drivers
            not_classified = pd.DataFrame(columns=df.columns)

        df['is_classified'] = True # Set all drivers from the main table as classified

        not_classified['finishing_status'] = 11  # TODO: should clean up the code later
        not_classified['is_classified'] = False

        df = pd.concat([df, not_classified], ignore_index=True)

        # Set col. names
        del df['NAT']
        df.columns = [i.upper() for i in df.columns]  # All col. names to upper case in case OCR
        df = df.rename(columns={                      # gives lower case
            'POSITION': 'finishing_position',
            'NO':       'car_no',
            'DRIVER':   'driver',
            'ENTRANT':  'team',
            'LAPS':     'laps_completed',
            'TIME':     'time',            # How long it took the driver to finish the race
            'GAP':      'gap',
            'INT':      'int',
            'KM/H':     'avg_speed',
            'FASTEST':  'fastest_lap_time',
            'ON':       'fastest_lap_no',  # The lap number on which the fastest lap was set
            'PTS':      'points'
        })
        df = df.replace({'': None})  # Empty string --> `None`, so `pd.isnull` works
        df.columns = [i.lower() for i in df.columns]

        # Clean up finishing status, e.g. is lapped? Is DSQ?
        df.loc[df.gap.fillna('').str.contains('LAP', regex=False), 'finishing_status'] = 1
        df.loc[(df.finishing_position == 'DNF') | (df.gap == 'DNF'), 'finishing_status'] = 11
        df.loc[(df.finishing_position == 'DQ') | (df.gap == 'DQ'), 'finishing_status'] = 20
        df.loc[(df.finishing_position == 'DNS') | (df.gap == 'DNS'), 'finishing_status'] = 30
        # TODO: clean up the coding
        # TODO: check how the PDF labels DQ? In the position col. or in the GAP col.? 2023 vs 2024

        # Add finishing position for DNF and DSQ drivers
        """
        For "usual" finishes, it's the finishing position in the PDF. If it's DNF, then put them
        after the usual finishes, and the relative order of DNF's is the same as their order in the
        PDF. Finally, DSQ drivers are in the end. Their relative order is also the same as in the
        PDF.
        """
        df.loc[df.finishing_position != 'DQ', 'temp'] = df.finishing_position
        df.temp = df.temp.astype(float)
        df.loc[df.finishing_position != 'DQ', 'temp'] \
            = df.loc[df.finishing_position != 'DQ', 'temp'].ffill() \
            + df.loc[df.finishing_position != 'DQ', 'temp'].isna().cumsum()
        df = df.sort_values(by='temp')
        df.temp = df.temp.ffill() + df.temp.isna().cumsum()
        df.finishing_position = df.temp.astype(int)
        del df['temp']

        df.car_no = df.car_no.astype(int)
        df.laps_completed = df.laps_completed.fillna(0).astype(int)
        df.time = df.time.apply(duration_to_millisecond)
        # TODO: gap to the leader is to be cleaned later, so we can use it for cross validation
        # TODO: is the `.fillna(0)` safe? See 2024 Brazil race Hulkenberg

        # Rank fastest laps
        """
        TODO: these need some serious cleaning.

        1. handling missing values, e.g. crash on/before finishing lap 1, so there is no fastest
           lap time to begin with
        2. proper ranking: currently we rank by lap time. If same, then rank by lap No. What if
           multiple drivers all set a same fastest lap time on the same lap? Need to combine the
           precise lap finishing calendar time (from other PDFs) with the lap time to rank them
           properly
        """
        # df.fastest_lap_time = pd.to_timedelta(df.fastest_lap_time)
        df.fastest_lap_no = df.fastest_lap_no.astype(float)
        df['fastest_lap_rank'] = df \
            .sort_values(by=['fastest_lap_time', 'fastest_lap_no'], ascending=[True, True]) \
            .groupby('car_no', sort=False) \
            .ngroup() + 1

        # Fill in some default values
        df = df.fillna({'points': 0, 'finishing_status': 0})
        df.finishing_status = df.finishing_status.astype(int)

        # Merge in starting grid from lap chart PDF
        if self.is_pdf_complete:
            df = df.merge(self.starting_grid, on='car_no', how='left')
        else:
            df['starting_grid'] = None

        def to_json() -> list[dict]:
            return df.apply(
                lambda x: SessionEntryImport(
                    object_type='SessionEntry',
                    foreign_keys=SessionEntryForeignKeys(
                        year=self.year,
                        round=self.round_no,
                        session=self.session,
                        car_number=x.car_no
                    ),
                    objects=[
                        SessionEntryObject(
                            position=x.finishing_position,
                            is_classified=x.is_classified,
                            status=x.finishing_status,
                            points=x.points,
                            time=x.time,
                            laps_completed=x.laps_completed,
                            fastest_lap_rank=x.fastest_lap_rank if x.fastest_lap_time else None,
                            grid=x.starting_grid
                            # TODO: replace the rank with missing or -1 in self.classification_df
                        )
                    ]
                ).model_dump(exclude_none=True, exclude_unset=True),
                axis=1
            ).tolist()

        df.to_json = to_json
        return df

    def _parse_history_chart(self) -> pd.DataFrame:
        doc = pymupdf.open(self.history_chart_file)
        dfs = []
        for page in doc:
            # Each page can have multiple (usually five) tables, all of which begins from the same
            # top y-position. The table headers are vertically bounded between "History Chart" and
            # "TIME"
            page = Page(page, file=self.history_chart_file)  # noqa: PLW2901
            history_chart = page.search_for('History Chart')
            if len(history_chart) != 1:
                raise ParsingError(f'Find none or multiple "History Chart" on p.{page.number} in '
                                   f'{self.history_chart_file}: {history_chart}')
            b_history_chart = history_chart[0].y1
            table_header = page.search_for('TIME', clip=(0, b_history_chart, page.w, page.h))
            if len(table_header) == 0:
                raise ParsingError(f'Could not find "TIME" on p.{page.number} in '
                                   f'{self.history_chart_file}: {table_header}')
            b_table_header = table_header[0].y1
            laps = page.search_for('Lap', clip=(0, b_history_chart, page.w, b_table_header))

            # Iterate over each table header and get the table content
            for i, lap in enumerate(laps):
                # Find table right boundary, which is the left of the next table header, or the
                # rightmost point of the page if it's the last table on the page
                if i + 1 < len(laps):
                    l_next_lap = laps[i + 1].x0 - 1
                else:
                    # At most five tables on a page, so one table at most has 20% of the page width
                    l_next_lap = min(page.w, lap.x0 + 0.2 * page.w)

                # Find table bottom, which is a white strip below the table header
                white_strip = page.search_for_white_strip(clip=(0, lap.y1, l_next_lap, page.h))
                if len(white_strip) == 0:
                    raise ParsingError(f'Could not find white strip below table {i} on p.'
                                       f'{page.number} in {self.history_chart_file}')
                b_table = white_strip[0]

                # Find the black line which separates the table header and table content
                l_table = lap.x0 - 1
                black_line = page.search_for_black_line(
                    clip=(l_table, lap.y1, l_next_lap, b_table),
                    min_length=0.7  # The line does not span across the entire clip area so shorter
                )                   # threshold
                if len(black_line) != 1:
                    raise ParsingError(
                        f'Could not find or find multiple black lines below the table header of '
                        f'table {i} on p.{page.number} in {self.history_chart_file}: {black_line}'
                    )
                t_table_body = black_line[0]

                # Get table header/col. names
                cols = self._detect_cols(
                    page,
                    clip=(l_table, b_history_chart, l_next_lap, t_table_body),
                    col_min_gap=2,  # Col. names are quite far from each other, so use a larger gap
                    min_black_line_length=0.6
                )
                if len(cols) != 3:  # noqa: PLR2004
                    raise ParsingError(f'Expected 3 cols. in the table {i} on p.{page.number} in '
                                       f'{self.history_chart_file}. Found: {cols}')
                if 'LAP' not in cols[0].text:
                    raise ParsingError(f'Expected "LAP" in the zero-th col. name of table {i} on '
                                       f'p.{page.number} in {self.history_chart_file}. Found: '
                                       f'{cols[0]}')
                lap_no = re.search(r'LAP (\d+)', cols[0].text)
                if not lap_no:
                    raise ParsingError(f'Expected "LAP x" to be the zero-th col. of table {i} on '
                                       f'p.{page.number} in {self.history_chart_file}. Found: '
                                       f'{cols[0]}')
                lap_no = int(lap_no.group(1))
                if cols[1].text != 'GAP':
                    raise ParsingError(f'Expected "GAP" to be the first col. of table {i} on '
                                       f'p.{page.number} in {self.history_chart_file}. Found: '
                                       f'{cols[1]}')
                if cols[2].text != 'TIME':
                    raise ParsingError(f'Expected "TIME" to be the second col. of table {i} on '
                                       f'p.{page.number} in {self.history_chart_file}. Found: '
                                       f'{cols[2]}')
                line_height = np.mean([i.b - i.t for i in cols])

                # Set col. separating vertical lines
                """
                We can use the left of "LAP x", the right of "LAP x", the midpoint between "GAP"
                and "TIME", and the left of "LAP x + 1" as col. separators. But I'm not very
                confident about the midpoint between "GAP" and "TIME", especially when the gap
                between two cars are big, or a lap time is very slow (e.g., due to accident then
                lingering into the pit). Therefore, I use `._detect_cols()` to detect the cols.
                This method is not designed for this purpose, but it does the job, as it sorts all
                words from left to right, and groups them into cols. based on their x-coord, so it
                works regardless of the y-coords.
                """
                temp = self._detect_cols(page,
                                         clip=(l_table, b_history_chart, l_next_lap, b_table),
                                         col_min_gap=1.1,
                                         min_black_line_length=0.6)
                if len(temp) != 3:  # noqa: PLR2004
                    raise ParsingError(f'Expected 3 cols. in the table {i} on p.{page.number} in '
                                       f'{self.history_chart_file}. Found: {temp}')
                vlines = [
                    l_table,
                    (temp[0].r + temp[1].l) / 2,
                    (temp[1].r + temp[2].l) / 2,
                    l_next_lap
                ]

                # Get row positions by checking white/grey background colours
                hlines = page.search_for_grey_white_rows(
                    clip=(l_table, t_table_body, l_next_lap, b_table),
                    min_height=line_height / 3
                )

                # Parse the table using the grid above
                df = self._parse_table_by_grid(page=page, vlines=vlines, hlines=hlines,
                                               header_included=False)
                df.columns = [i.text.split()[0].lower() for i in cols]  # "LAP 5" --> "lap"
                df = df.rename(columns={'lap': 'car_no'})
                df['lap'] = lap_no
                df = df[df.car_no != '']  # Sometimes will get one additional empty row

                # The row order/index is meaningful: it's the order/positions of the cars
                # Need extra care for lapped cars (harningle/fia-doc#19)
                # TODO: is this true for all cases? E.g. retirements?
                df = df.reset_index(drop=False, names=['position'])
                df['position'] += 1  # 1-indexed
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        del dfs
        df.car_no = df.car_no.astype(int)

        # Extra cleaning for lapped cars
        """
        There is one tricky place: when a car is lapped (GAP col. is "1 LAP" or more), the actual
        lap number for the lapped car should be the lap number in PDF minus the #. of laps being
        lapped. That is, when the leader starts lap 10 (the table header is lap 10), the lapped car
        starts his lap 9 if GAP is "1 LAP".

        The lapping itself is easy to fix, but when a lapped car is in pit stop, the PDF shows
        "PIT" in the GAP col., so we cannot distinguish between a normal car in pit versus a lapped
        car in pit, and as a result we cannot fix the lap number for the lapped car. After applying
        the above fix, we will get duplicated lap numbers for a lapped car if it pits after being
        lapped. We shift the lap number for the lapped car by 1 to get the correct lap number. See
        the below example: we have lap number 30, 31, 33, 33 and it should be 30, 31, 32, 33. We
        shift the first "33" to "32" to fix it.

        in PDF              before fix      after fix "1 LAP"   after fix "PIT"

        LAP 31              lap time        lap time            lap time
        1 LAP   1:39.757    31  1:39.757    30  1:39.757        30  1:39.757
                                            ↑↑

        LAP 32
        1 LAP   1:39.748    32  1:39.748    31  1:39.748        31  1:39.748
                                            ↑↑

        LAP 33
        PIT     1:44.296    33  1:44.296    33  1:44.296        32  1:44.296
                                                                ↑↑

        LAP 34
        PIT     2:18.694    34  2:18.694    33  2:18.694        33  2:18.694
                                            ↑↑

        TODO: is this really mathematically correct? Can a lapped car pits and then gets unlapped?
        """
        df.lap = df.lap - df.gap.apply(
            lambda x: int(re.findall(r'\d+', x)[0]) if 'LAP' in x else 0
        )
        df = df.reset_index(drop=False).sort_values(by=['car_no', 'lap', 'index'])
        df.loc[(df.car_no == df.car_no.shift(-1)) & (df.lap == df.lap.shift(-1)), 'lap'] -= 1
        df.loc[(df.car_no == df.car_no.shift(1)) & (df.lap == df.lap.shift(1) + 2), 'lap'] -= 1
        del df['index']

        # TODO: Perez "retired and rejoined" in 2023 Japanese... Maybe just mechanically assign lap
        #       No. as 1, 2, 3, ... for each driver?
        return df

    def _parse_lap_chart(self) -> pd.DataFrame:
        doc = pymupdf.open(self.lap_chart_file)
        dfs = []
        for page in doc:
            page = Page(page, file=self.lap_chart_file)  # noqa: PLW2901

            # Table header/col. names are below "Race Lap Chart"
            b_lap_chart = page.search_for('Lap Chart')[0].y1

            # Table header is above a black line
            black_line = page.search_for_black_line(clip=(0, b_lap_chart, page.w, page.h))
            if not black_line:  # Maybe two lines: one for table top and one for page bottom
                raise ParsingError(f'Could not find the black line below the table header '
                                   f'on p.{page.number} in {self.lap_chart_file}')
            t_table_body = black_line[0]

            # Table bottom is a white strip
            """
            A white strip of any height is fine. Because the table always has a black vertical line
            between col. 0 and col. 1, so there is no white strip at all in the table. Any white
            strip must indicate the end of the table.
            """
            white_strip = page.search_for_white_strip(clip=(0, t_table_body, page.w, page.h),
                                                      height=1)
            if not white_strip:
                raise ParsingError(f'Could not find any white strip below the table on '
                                   f'p.{page.number} in {self.lap_chart_file}')
            b_table = white_strip[0]

            # Get col. names between the two y-coords. above
            cols = self._detect_cols(page,
                                     clip=(0, b_lap_chart + 1, page.w, t_table_body - 1),
                                     col_min_gap=3,  # Col. names are quite far from each other
                                     min_black_line_length=0.5)
            if len(cols) <= 1:
                raise ParsingError(f'Expected at least two cols. in table on p.{page.number} in '
                                   f'{self.lap_chart_file}. Found: {cols}')
            if cols[0].text != 'POS':
                raise ParsingError(f'Expected "POS" to be the zero-th col. on p.{page.number} '
                                   f'in {self.lap_chart_file}. Found: {cols[0]}')
            for i in range(1, len(cols)):
                if not re.match(r'^\d+$', cols[i].text):
                    raise ParsingError(
                        f'Expected the {i}-th col. to be a number on p.{page.number} in '
                        f'{self.lap_chart_file}. Found: {cols[i]}'
                    )

            # Find a black vertical line below the black horizontal line above. This separates the
            # zero-th col. and the first col.
            """
            `Page.search_for_black_line` is for horizontal lines only. So to search for vertical
            lines, we rotate the page, i.e. applying [[0, -1], [1, 0]].
            """
            page.set_rotation(270)
            black_line = page.search_for_black_line(
                clip=(t_table_body, 0, b_table, page.w),    # `clip` is rotated too
                min_length=0.7
            )
            black_line = [page.w - i for i in black_line]  # Transpose back
            page.set_rotation(0)
            if len(black_line) != 1:
                raise ParsingError(f'Could not find or find multiple vertical black lines below '
                                   f'the table header on p.{page.number} in '
                                   f'{self.lap_chart_file}: {black_line}')
            l_first_col = black_line[0]
            vlines = [
                0,
                l_first_col,
                *[(cols[i].r + cols[i + 1].l) / 2 for i in range(1, len(cols) - 1)],
                page.w
            ]

            # Locate rows. Same intuition as getting col. positions in `._parse_history_chart()`
            rows = self._detect_rows(page,
                                     clip=(0, t_table_body, l_first_col - 2, b_table),
                                     min_black_line_length=0.5)
            if not rows:
                raise ParsingError(f'Could not detect any rows in the zero-th col. in table on '
                                   f'p.{page.number} in {self.lap_chart_file}')
            for row in rows:
                if not re.match(r'^GRID$|^LAP \d+$', row.text):
                    raise ParsingError(f'Expected "GRID" or "LAP x" for all cells in the zero-th '
                                       f'col. in table on p.{page.number} in '
                                       f'{self.lap_chart_file}. Found: {row}')
            hlines = [
                t_table_body,
                *[(rows[i].b + rows[i + 1].t) / 2 for i in range(len(rows) - 1)],
                b_table
            ]

            # Parse the table
            df = self._parse_table_by_grid(
                page=page,
                vlines=vlines,
                hlines=hlines,
                header_included=False
            )

            # Reshape to long format, where a row is (lap, driver, position)
            df.columns = [i.text for i in cols]
            df.index.name = None
            if (df.POS == 'GRID').any():
                self.starting_grid = (df[df.POS == 'GRID']
                                      .drop(columns='POS')
                                      .T
                                      .reset_index()
                                      .rename(columns={'index': 'starting_grid', 0: 'car_no'}))
                self.starting_grid.car_no = self.starting_grid.car_no.astype(int)
                self.starting_grid.starting_grid = self.starting_grid.starting_grid.astype(int)
                df = df[df.POS != 'GRID']
            df.POS = df.POS.str.removeprefix('LAP ').astype(int)
            df = (df.set_index('POS')
                  .stack()
                  .reset_index(name='car_no')
                  .rename(columns={'POS': 'lap', 'level_1': 'position'}))
            df = df[df.car_no.notna() & (df.car_no != '')]  # E.g., a car retires on lap 10, so lap
            df.position = df.position.astype(int)           # 11 will have a missing car No.
            df.car_no = df.car_no.astype(int)
            dfs.append(df)
        return pd.concat(dfs, ignore_index=True)

    def _parse_lap_analysis(self) -> pd.DataFrame:
        doc = pymupdf.open(self.lap_analysis_file)
        dfs = []
        for page in doc:
            # Find "Race Lap Analysis"
            page = Page(page, file=self.lap_analysis_file)  # noqa: PLW2901
            race_lap_analysis = page.search_for('Lap Analysis')  # Can be "Sprint Lap Analysis" so
            if len(race_lap_analysis) != 1:                      # only search for "Lap Analysis"
                raise ParsingError(f'Find none or multiple "Lap Analysis" on p.{page.number} in '
                                   f'{self.lap_analysis_file}')
            b_lap_analysis = race_lap_analysis[0].y1

            # Find the white strips immediately below "Lap Analysis". Below this are the tables
            white_strip = page.search_for_white_strip(clip=(0, b_lap_analysis, page.w, page.h))
            if not white_strip:
                raise ParsingError(f'Expect at least a white strip below "Lap Analysis" on '
                                   f'p.{page.number} in {self.lap_analysis_file}. Found: '
                                   f'{white_strip}')
            t_all_drivers = white_strip[0]  # Driver names are between the first two white strips

            # Find all black horizontal lines
            """
            To be precise, it's multiple black line segments. The PDF usually has three tables side
            by side. All tables have two black lines separating the header and table body. And all
            these lines have the same y-position. So we have six line segments, and they will be
            detected as one black line, because we find lines by looking at rows with enough amount
            of black pixels. That's fine. We don't need the x-positions of the lines. Knowing the
            y-coord. of these lines is good enough for us to locate the table body.

            Depending on the number of drivers on the page, we can have two line segments to six.
            So the total length of all the lines can be very short or long. But the minimum is two
            black line segments, which should easily be more than 25% of the page width. So we set
            `min_length=0.25` below.
            """
            black_lines = page.search_for_black_line(clip=(0, t_all_drivers, page.w, page.h),
                                                     min_length=0.25)
            if not black_lines:
                raise ParsingError(f'Could not find any black line below "Lap Analysis" on '
                                   f'p.{page.number} in {self.lap_analysis_file}')

            # Each line should be a separator between the table header and body
            t_table_headers = []
            t_drivers = []
            b_tables = []
            for i in range(len(black_lines) - 1, -1, -1):
                # The table header is vertically between the black line and the white strip
                # immediately above the black line
                black_line = black_lines[i]
                white_strip = page.search_for_white_strip(clip=(0, 0, page.w, black_line))
                if not white_strip:
                    raise ParsingError(f'Could not find any white strips above the black line '
                                       f'at {black_line} on p.{page.number} in '
                                       f'{self.lap_analysis_file}. Found: {white_strip}')
                t_table_header = sorted(white_strip)[-1]
                header = page.get_text(clip=(0, t_table_header, page.w, black_line))

                # If no table header found, then it's the last black line at the bottom of the
                # page. Drop it
                if not ('LAP' in header and 'TIME' in header):
                    black_lines.pop(i)
                    continue
                t_table_headers.insert(0, t_table_header)

                # The driver name is above the table header and the next white strip above
                if len(white_strip) < 2:  # noqa: PLR2004
                    raise ParsingError(f'Expect at least two white strips above the black line at '
                                       f'{black_line} on p.{page.number} in '
                                       f'{self.lap_analysis_file}. Found: {white_strip}')
                t_drivers.insert(0, white_strip[-2])

                # Table bottom is the next white strip below the black line
                white_strip = page.search_for_white_strip(clip=(0, black_line, page.w, page.h))
                if not white_strip:
                    raise ParsingError(f'Could not find any white strip below the black line at '
                                       f'{black_line} on p.{page.number} in '
                                       f'{self.lap_analysis_file}')
                b_tables.insert(0, white_strip[0])

            # Two tables are vertically separated by a vertical white strip
            page.set_rotation(90)
            table_separators = page.search_for_white_strip(
                clip=(page.h - b_tables[-1], 0, page.h - t_drivers[0], page.w),
                height=0.03 * page.w  # White strip should occupy at least 3% of the page width
            )
            page.set_rotation(0)
            # Should have at least one table, so two white strips to the left and right of it
            if len(table_separators) < 2:  # noqa: PLR2004
                raise ParsingError(f'Expect at least two vertical white strips below driver names '
                                   f'on p.{page.number} in {self.lap_analysis_file}. Found: '
                                   f'{table_separators}')

            # Loop through each table
            for i in range(len(t_drivers)):
                t_driver = t_drivers[i] + 1
                t_table_header = t_table_headers[i] + 1
                b_table_header = black_lines[i] - 1
                b_table = b_tables[i] + 1
                for j in range(0, len(table_separators) - 1):
                    l_table = max(0, table_separators[j] - 1)
                    r_table = table_separators[j + 1] + 1

                    # Get the driver name and car No.
                    driver = page.get_text(clip=(l_table, t_driver, r_table, t_table_header))
                    if not driver.strip():  # E.g., four tables on a page. The second row only has
                        continue            # one table, so we will have missing's here
                    car_no = driver.split('\n', 1)[0]
                    try:
                        car_no = int(car_no.strip())
                    except:  # noqa: E722
                        raise ParsingError(f'Could not parse car No. in '
                                           f'({l_table:.2f}, {t_driver:.2f}, {r_table:.2f}, '
                                           f'{t_table_header:.2f}) on p.{page.number} in '
                                           f'{self.lap_analysis_file}: {driver}')

                    # Find the vertical white strip separating the two tables for the driver
                    page.set_rotation(90)
                    white_strip = page.search_for_white_strip(
                        clip=(page.h - b_table, l_table, page.h - t_table_header, r_table),
                        height=1  # Any height is fine
                    )
                    page.set_rotation(0)
                    # Table left, separator, and right. Three in total
                    if len(white_strip) != 3:  # noqa: PLR2004
                        raise ParsingError(f'Expected exactly three vertical white strips in '
                                           f'({l_table:.2f}, {t_table_header:.2f}, '
                                           f'{r_table:.2f}, {b_table:.2f}) on p.'
                                           f'{page.number} in {self.lap_analysis_file}. '
                                           f'Found: {white_strip}')
                    m_table = white_strip[1] + 1

                    # Parse each of the two tables of the driver
                    for l_tab, r_tab in [(l_table, m_table), (m_table, r_table)]:
                        # Refine table bottom
                        """
                        `b_table` above is the bottom of all six table bottoms on the page. E.g., a
                        driver finishes the race and has 60 laps or 60 rows in his table. Another
                        driver crashes out after 10 laps, so his table only has 10 rows. The bottom
                        of the two tables are different. Therefore, here we get a more accurate
                        table bottom for each table.
                        """
                        white_strip = page.search_for_white_strip(
                            clip=(l_tab, b_table_header, r_tab, page.h)
                        )
                        if not white_strip:
                            raise ParsingError(f'Could not find any white strip below the table in '
                                               f'({l_tab:.2f}, {b_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table:.2f}) on p.'
                                               f'{page.number} in {self.lap_analysis_file}')
                        b_table = white_strip[0]

                        # Get table header
                        cols = self._detect_cols(
                            page,
                            clip=(l_tab, t_table_header, r_tab, b_table_header),
                            col_min_gap=3,
                            min_black_line_length=0.5
                        )
                        if len(cols) != 2:  # noqa: PLR2004
                            raise ParsingError(f'Expected exactly two cols. in '
                                               f'({l_tab:.2f}, {t_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table_header:.2f}) on p.'
                                               f'{page.number} in {self.lap_analysis_file}. '
                                               f'Found: {cols}')
                        if cols[0].text != 'LAP' or cols[1].text != 'TIME':
                            raise ParsingError(f'Expected "LAP" and "TIME" to be the two col. '
                                               f'names in ({l_tab:.2f}, {t_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table_header:.2f}) on p.'
                                               f'{page.number} in {self.lap_analysis_file}. '
                                               f'Found: {cols}')
                        l_tab = cols[0].l - 1  # More accurate table left boundary  # noqa: PLW2901

                        # Vertical lines separating the two cols.
                        vlines = [l_tab, (cols[0].r + cols[1].l) / 2, r_tab]

                        # Horizontal lines are located by the grey and white rows
                        hlines = page.search_for_grey_white_rows(
                            clip=(l_tab, b_table_header, r_tab, b_table),
                            min_height=np.mean([i.b - i.t for i in cols]) - 1
                        )

                        # Parse the table
                        df = self._parse_table_by_grid(page=page,
                                                       vlines=vlines,
                                                       hlines=hlines,
                                                       header_included=False)
                        if df.empty:  # E.g., DNS so no lap at all
                            continue
                        df.columns = [i.text.lower() for i in cols]

                        # TODO: testing on the cols. `lap` should has digits and "P", and `time`
                        #       should be a time duration format

                        df['pit'] = df.lap.str.contains('P', regex=False)
                        df.lap = df.lap.str.rstrip(' P').astype(int)
                        df['car_no'] = int(car_no)
                        dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def _parse_lap_times(self) -> pd.DataFrame:
        # Get lap times from Race Lap Analysis PDF
        df = self._parse_lap_analysis()
        df = df.rename(columns={'time': 'lap_time'})

        # Lap 1's lap times are calendar time in Race Lap Analysis. To get the actual lap time for
        # lap 1, we parse Race History Chart PDF
        lap_1 = (self._parse_history_chart()[['car_no', 'lap', 'time']]
                 .sort_values(by=['car_no', 'lap'])
                 .groupby('car_no')
                 .first()  # See #60
                 .assign(lap=1)
                 .reset_index())
        df = df.merge(lap_1, on=['car_no', 'lap'], how='outer', indicator=True, validate='1:1')
        assert (df[df.lap == 1]['_merge'] == 'both').all(), \
            f"Lap 1's data do not match in {self.lap_analysis_file} and {self.history_chart_file}"
        df.loc[df.lap == 1, 'lap_time'] = df.loc[df.lap == 1, 'time']
        del df['time'], df['_merge'], lap_1

        # Merge in car positions from Race Lap Chart PDF
        positions = self._parse_lap_chart()
        df = df.merge(positions, on=['car_no', 'lap'], how='outer', indicator=True, validate='1:1')
        assert (df._merge == 'both').all(), f'Some laps only found in only one of ' \
                                            f'{self.lap_analysis_file} and {self.lap_chart_file}'
        del df['_merge'], positions

        # Merge in the fastest lap info. from final classification
        # TODO: drivers DNS or DNF before end of lap 1 have no lap at all, so drop them. Check
        temp = self.classification_df[['car_no', 'fastest_lap_time', 'fastest_lap_no',
                                       'laps_completed']]
        temp = temp[temp.laps_completed >= 1]
        df = df.merge(temp, on='car_no', how='outer', indicator=True, validate='m:1')
        # TODO: I really want the below check but it fails to handle one case: a driver drives
        #       normally but get DSQ after several laps. He will have lap data in the lap analysis
        #       PDF but in classification PDF, he is DSQ so has zero lap completed. In such case,
        #       we will have more drivers in the lap analysis PDF than in the classification PDF,
        #       and the below check will fail
        # assert (df._merge == 'both').all(), \
        #     f'Some drivers only found in only one of {self.lap_analysis_file} and ' \
        #     f'{self.classification_file}: {df[df._merge != "both"].car_no.unique()}'
        del df['_merge']
        temp = df[df.lap == df.fastest_lap_no]
        if (temp.lap_time != temp.fastest_lap_time).any():
            diff = temp[temp.lap_time != temp.fastest_lap_time]
            raise AssertionError(f'fastest lap time in lap times PDF does not match the one in '
                                 f'classification PDF\n: {diff.to_string(index=False)}')
        df['is_fastest_lap'] = df.lap == df.fastest_lap_no
        del df['fastest_lap_time'], df['fastest_lap_no']

        def to_json() -> list[dict]:
            temp = df.copy()
            temp.lap = temp.apply(
                lambda x: LapObject(
                    number=x.lap,
                    position=x.position,
                    time=duration_to_millisecond(x.lap_time),
                    is_entry_fastest_lap=x.is_fastest_lap
                ),
                axis=1
            )
            temp = temp.groupby('car_no')[['lap']].agg(list).reset_index()
            temp['session_entry'] = temp.car_no.map(
                lambda x: SessionEntryForeignKeys(
                    year=self.year,
                    round=self.round_no,
                    session='R' if self.session == 'race' else 'SR',
                    car_number=x
                )
            )
            return temp.apply(
                lambda x: LapImport(
                    object_type="Lap",
                    foreign_keys=x.session_entry,
                    objects=x.lap
                ).model_dump(exclude_unset=True),
                axis=1
            ).tolist()

        df.to_json = to_json
        return df

    def _cross_validate(self) -> None:
        """Cross validate against other PDFs or fastf1?"""
        raise NotImplementedError


class QualifyingParser(BaseParser):
    # TODO: need better docstring
    # TODO: probably need to refactor this. Not clean
    def __init__(
            self,
            classification_file: str | os.PathLike,
            lap_times_file: Optional[str | os.PathLike],
            year: int,
            round_no: int,
            session: QualiSessionT
    ):
        self.classification_file = classification_file
        self.lap_times_file = lap_times_file
        self.session = session
        self.year = year
        self.round_no = round_no
        self._check_session()
        # self._cross_validate()

    @cached_property
    def is_pdf_complete(self) -> bool:
        # TODO: need this property?
        if self.lap_times_file is None:
            return False
        return True

    @cached_property
    def classification_df(self) -> pd.DataFrame:
        return self._parse_classification()

    @cached_property
    def lap_times_df(self) -> pd.DataFrame:
        if not self.is_pdf_complete:
            warnings.warn('Lap times PDF is missing. Can get fastest laps only from the '
                          'classification PDF')
            df = self._apply_fallback_fastest_laps(pd.DataFrame(columns=['car_no'], data=[]),
                                                   self.classification_df.NO.unique())
            df.to_json = partial(quali_lap_times_to_json, df=df,
                                 year=self.year, round_no=self.round_no, session=self.session)
            return df
        else:
            return self._parse_lap_times()

    def _check_session(self) -> None:
        """Check that the input session is valid. Raise an error otherwise"""
        if self.session not in get_args(QualiSessionT):
            raise ValueError(f'Invalid session: {self.session}. '
                             f'Valid sessions are: {get_args(QualiSessionT)}"')
        return

    def _parse_classification(self):
        # Find the page with "Qualifying Session Final Classification"
        doc = pymupdf.open(self.classification_file)
        classification = []
        for i in range(len(doc)):
            page = Page(doc[i], file=self.classification_file)
            if '.pdf' in page.get_text():  # Fix #59
                continue
            classification = page.search_for('Final Classification')
            if classification:
                break
            classification = page.search_for('Provisional Classification')
            if classification:
                warnings.warn('Found and using provisional classification, not the final one')
                break
        if not classification:
            doc.close()  # TODO: check docs. Do we need to manually close it? Memory safe?
            raise ValueError(f'"Final Classification" or "Provisional Classification" not found '
                             f'on any page in {self.classification_file}')

        # Bottom of "Final Classification"
        b_classification = classification[0].y1 + 1

        # First black horizontal line below "Final Classification"
        if black_line := page.search_for_black_line(clip=(0, b_classification, page.w, page.h)):
            t_table_body = black_line[0]
        else:
            raise ParsingError(f'Cannot find the black line separating table header and table '
                               f'body below "Final Classification" on p.{page.number} in '
                               f'{self.classification_file}')

        # Get text height of the table header row
        temp = page.get_text('blocks', clip=(0, b_classification, page.w, t_table_body))
        if not temp:
            raise ParsingError(f'Cound not find any text in the table header on p.{page.number} '
                               f'in {self.classification_file}')
        line_height = np.mean([i[3] - i[1] for i in temp])

        # Find the first white strip below the table header
        if white_strip:= page.search_for_white_strip(clip=(0, t_table_body, page.w, page.h),
                                                     height=line_height / 3):
            b_table = white_strip[0]
        else:
            raise ParsingError(f'Could not find table bottom by white strip on p.{page.number} in '
                               f'{self.classification_file}')

        # Get col. names
        col_names = self._detect_cols(page,
                                      clip=(0, b_classification, page.w, t_table_body - 1),
                                      col_min_gap=1)
        if not col_names:
            raise ParsingError(f'Could not detect cols. in the table header on p.{page.number} in '
                               f'{self.classification_file}')
        col_names.sort(key=lambda x: x.l)  # Sort cols. from left to right
        if self.session == 'sprint_quali':  # Always use "Q1", "Q2", and "Q3" for both quali. and
            for col in col_names:           # sprint quali.
                if col.text == 'SQ1':
                    col.text = 'Q1'
                elif col.text == 'SQ2':
                    col.text = 'Q2'
                elif col.text == 'SQ3':
                    col.text = 'Q3'
        for col_name in ['LAPS', 'TIME']:  # Add prefix to "LAPS" and "TIME" for each session
            session = 1
            for col in col_names:
                if col.text == col_name:
                    col.text = f'Q{session}_{col.text}'
                    session += 1
        col_names = {i.text: i for i in col_names}

        # Get col. vertical positions
        """
        All cols. can easily be located by the left or right boundary of the col. name, except for
        the boundary between ENTRANT and Q1, Q1 % and Q1_TIME, Q1_TIME and Q2, and Q2_TIME and Q3.
        We cannot handle them using `._detect_cols` with the entire table area, because the gaps
        between cols. are tiny (NAT and ENTRANT are very close), so we need to use small
        `col_min_gap`. However, a small `col_min_gap` will split some cols. into two (e.g.,
        1:23.456 into 1: and 23.456). Therefore, we handle these three boundaries separately.

        ENTRANT and Q1:
        We look at the table body area between the left of ENTRANT and the left of Q1_LAPS. This
        area should contain the ENTRANT col. and the Q1 col. only. Between them is some sizable
        gap. We can use the gap as the boundary between the two cols.

        Q1 % and Q1_TIME:
        We don't always have the Q1 % col. (the "%" as in the 107% rule). E.g., 2024 Chinese has
        this col., but 2024 Las Vegas does not. If we do have it in `col_names`, then do it.
        Otherwise all good.

        Q1_TIME and Q2, and Q2_TIME and Q3:
        There is a vertical thin line between them. The line is super thin so we need to use higher
        DPI (bigger `scaling_factor`) to detect it.
        """
        # Boundary between ENTRANT and Q1
        cols = self._detect_cols(
            page,
            clip=(
                col_names['ENTRANT'].l - 1,
                t_table_body + 1,
                col_names['Q1_LAPS'].l,
                b_table + 1
            ),
            col_min_gap=1
        )
        if not (len(cols) == 2 and re.match(r'[\d:\n.]+', cols[1].text)):  # noqa: PLR2004
            raise ParsingError(f'Could not locate the boundary between ENTRANT and Q1 on '
                               f'p.{page.number} in {self.classification_file}: {cols}')
        sep_entrant_q1 = (cols[0].r + cols[1].l) / 2
        # Boundary between Q1 % and Q1_TIME
        if '%' in col_names:
            cols = self._detect_cols(
                page,
                clip=(
                    col_names['%'].l,
                    t_table_body + 1,
                    col_names['Q1_TIME'].l,
                    b_table + 1
                ),
                col_min_gap=1.1
            )
            if not (len(cols) == 2  # noqa: PLR2004
                    and re.match(r'[\d.]+', cols[0].text)
                    and re.match(r'[\d.]+', cols[1].text)):
                raise ParsingError(f'Could not locate the boundary between Q1 % and Q1_TIME on '
                                   f'p.{page.number} in {self.classification_file}: {cols}')
            sep_q1_pct_q1time = (cols[0].r + cols[1].l) / 2
        # Boundary between Q1_TIME and Q2
        page.set_rotation(90)
        black_line = page.search_for_black_line(
            clip=(page.h - b_table,
                col_names['Q1_TIME'].r,
                page.h - t_table_body,
                col_names['Q2'].l
            ),
            scaling_factor=16
        )
        if len(black_line) != 1:
            raise ParsingError(f'Could not locate the boundary between Q1_TIME and Q2 on '
                               f'p.{page.number} in {self.classification_file}: {black_line}')
        sep_q1time_q2 = black_line[0]
        # Boundary between Q2_TIME and Q3
        black_line = page.search_for_black_line(
            clip=(
                page.h - b_table,
                col_names['Q2_TIME'].r,
                page.h - t_table_body,
                col_names['Q3'].l
            ),
            scaling_factor=16
        )
        if len(black_line) != 1:
            raise ParsingError(f'Could not locate the boundary between Q2_TIME and Q3 on '
                               f'p.{page.number} in {self.classification_file}: {black_line}')
        sep_q2time_q3 = black_line[0]
        page.set_rotation(0)
        # All col. positions
        vlines = [
            0,
            col_names['NO'].l - 1,
            (col_names['NO'].r + col_names['DRIVER'].l) / 2,
            col_names['NAT'].l - 1,
            (col_names['NAT'].r + col_names['ENTRANT'].l) / 2,
            sep_entrant_q1,
            col_names['Q1_LAPS'].l,
            col_names['Q1_LAPS'].r,
            sep_q1time_q2,
            col_names['Q2_LAPS'].l,
            col_names['Q2_LAPS'].r,
            sep_q2time_q3,
            col_names['Q3_LAPS'].l,
            col_names['Q3_LAPS'].r,
            page.w
        ]
        if '%' in col_names:
            # TODO: this assumes dict is ordered. Not sure which Python version started this
            vlines.insert(list(col_names.keys()).index('%') + 2, sep_q1_pct_q1time)

        # Row positions
        hlines = page.search_for_grey_white_rows(clip=(0, t_table_body, page.w, b_table),
                                                 min_height=line_height / 3)

        # Get the table
        df = self._parse_table_by_grid(page, vlines=vlines, hlines=hlines, header_included=False)
        df.columns = ['position'] + [i for i in col_names.keys()]
        df['finishing_status'] = 0
        df['original_order'] = range(1, len(df) + 1)  # Driver's original order in the PDF
        df['is_classified'] = True

        # Parse "NOT CLASSIFIED" table, if any
        if not_classified := page.search_for('NOT CLASSIFIED', clip=(0, b_table, page.w, page.h)):
            t_table_body = not_classified[0].y1
            if white_strip:= page.search_for_white_strip(clip=(0, t_table_body, page.w, page.h),
                                                         height=line_height / 3):
                b_table = white_strip[0]
            else:
                raise ParsingError(
                    f'Could not find the bottom of "NOT CLASSIFIED" table by white strip on '
                    f'p.{page.number} in {self.classification_file}'
                )
            hlines = page.search_for_grey_white_rows(clip=(0, t_table_body, page.w, b_table),
                                                     min_height=line_height / 3)
            not_classified = self._parse_table_by_grid(page, vlines=vlines, hlines=hlines,
                                                       header_included=False)
            not_classified.columns = df.columns.drop(['finishing_status', 'original_order',
                                                      'is_classified'])
            not_classified.position = None  # No finishing position for unclassified drivers
            n = len(df)
            not_classified['original_order'] = range(n + 1, n + len(not_classified) + 1)
            not_classified['finishing_status'] = 11  # TODO: should clean up the code later
            not_classified['is_classified'] = False
        else:
            not_classified = pd.DataFrame(columns=df.columns)
        df = pd.concat([df, not_classified], ignore_index=True)

        # Parse "DISQUALIFIED" table, if any
        if disqualified := page.search_for('DISQUALIFIED', clip=(0, b_table, page.w, page.h)):
            """
            There can be multiple "DISQUALIFIED" text. E.g., in the penalty notes, we may have
            "DISQUALIFIED", and we may have "DISQUALIFIED" as the table title. The table title
            should be horizontally centred on the page, so we do the following check.
            """
            disqualified.sort(key=lambda x: x.y0)  # The topmost "DISQUALIFIED"
            disqualified = disqualified[0]
            if np.isclose((disqualified.x0 + disqualified.x1) / 2, page.w / 2, rtol=0.1):
                t_table_body = disqualified.y1
                if white_strip := page.search_for_white_strip(
                        clip=(0, t_table_body, page.w, page.h),
                        height=line_height / 3
                ):
                    b_table = white_strip[0]
                else:
                    raise ParsingError(
                        f'Could not find the bottom of "DISQUALIFIED" table by white strip on '
                        f'p.{page.number} in {self.classification_file}'
                    )
                hlines = page.search_for_grey_white_rows(clip=(0, t_table_body, page.w, b_table),
                                                         min_height=line_height / 3)
                disqualified = self._parse_table_by_grid(page, vlines=vlines, hlines=hlines,
                                                         header_included=False)
                disqualified.columns = df.columns.drop(['finishing_status', 'original_order',
                                                        'is_classified'])
                disqualified.position = None  # No finishing position for DSQ drivers
                n = len(df)
                disqualified['original_order'] = range(n + 1, n + len(disqualified) + 1)
                disqualified['finishing_status'] = 20  # TODO: should clean up the code later
                disqualified['is_classified'] = False
            else:
                warnings.warn(f'Found "DISQUALIFIED" on p.{page.number} in '
                              f'{self.classification_file}, but it is not horizontally '
                              f'centred. May be a penalty note instead of a DISQUALIFIED table. '
                              f'Ignored')
                disqualified = pd.DataFrame(columns=df.columns)
        else:
            disqualified = pd.DataFrame(columns=df.columns)
        df = pd.concat([df, disqualified], ignore_index=True)

        """
        `is_classified` here is simply a flag to indicate whether the driver belongs to the "NOT
        CLASSIFIED" table. It doesn't mean a driver is classified or not in F1 sense.. Basically
        everyone in "NOT CLASSIFIED" table is not classified, and in addition, those in the main
        table who receive DSQ or DNQ are not classified either.
        """

        # Fill in the position for DNF and DSQ drivers
        # TODO: check this
        df.loc[df.position.isin(['DQ', 'DSQ']), 'finishing_status'] = 20
        df.loc[df.position != 'DQ', 'temp'] = df.position
        df.temp = df.temp.astype(float)
        df.loc[df.position != 'DQ', 'temp'] = df.loc[df.position != 'DQ', 'temp'].ffill() \
                                              + df.loc[df.position != 'DQ', 'temp'].isna().cumsum()
        df = df.sort_values(by='temp')
        df.temp = df.temp.ffill() + df.temp.isna().cumsum()
        df.position = df.temp.astype(int)
        del df['temp']

        # Clean up
        df = df.replace('', None)  # So pd.isna will catch empty string as well
        df.NO = df.NO.astype(int)
        del df['NAT']
        df.position = df.position.astype(int)

        # Overwrite `.to_json()` and `.to_pkl()` methods
        # TODO: bad practice
        def to_json() -> list[dict]:
            """
            We want to get the drivers and results in each session. E.g., for Q2, this means two
            things: (1) the top 15 drivers, and (2) other drivers that get non-empty laps in Q2.
            (1) and (2) are not necessarily the same. E.g., one driver can be in Q2 but doesn't do
            any lap (maybe to save tyre for the race), so his "LAPS" is empty in the PDF. Or a
            driver can do some laps in Q2 but he is outside top 15, e.g. 2025 Bahrain Hulkenberg
            went through Q1 and did some laps in Q2, but during the quali. his Q1 fastest lap was
            deleted, so he was not classified in Q2 so ranked 16th (#50).
            """
            data = []
            for q in [1, 2, 3]:
                n_drivers = QUALI_DRIVERS[self.year][q]
                temp = pd.concat([df[df.original_order <= n_drivers],
                                  df[(df.original_order > n_drivers) & df[f'Q{q}_LAPS'].notna()]])
                # Clean up DNS/DNF/DSQ drivers
                temp.loc[temp[f'Q{q}'].isin(['DQ', 'DSQ']), 'finishing_status'] = 20
                temp.loc[temp[f'Q{q}'] == 'DNF', 'finishing_status'] = 11
                temp.loc[temp[f'Q{q}'] == 'DNS', 'finishing_status'] = 30
                temp.loc[temp[f'Q{q}'].isin(['DNS', 'DNF']), f'Q{q}'] = 'Z'
                temp['is_dsq'] = (temp.finishing_status == 20)  # noqa: PLR2004
                temp['is_dnq'] = (temp.original_order > n_drivers)
                temp.loc[temp.is_dnq, 'finishing_status'] = 40
                """
                `is_classified` is defined following 2025 Formula 1 Sporting Regulations 39.4 b),
                published on 2025/02/26, available at https://www.fia.com/system/files/documents.
                A driver is not classified if any of the following is true:

                1. 107% rule not met, i.e. he is in the "NOT CLASSIFIED" table
                2. DSQ
                3. no valid lap is done, e.g. all laps are deleted for exceeding track limits

                1. is already done above when we parsing "NOT CLASSIFIED" table. 2. can be detected
                by looking at the finishing status. 3. is done by checking if the driver has a
                lap time in the table: if the lap time col. is not a time but something else, e.g.
                DNF, the driver does not set a valid time so he is not classified.
                """
                temp.loc[(temp.finishing_status != 0) & temp.is_classified,
                         'is_classified'] = False
                """
                There are four types of drivers: (1) finishing normally, (2) DNF or DNS, (3) DSQ,
                and (4) DNQ. (1) will come first, and within (1) we sort by their fastest lap time.
                (2) are drivers who participate the session normally, but do not get a lap time,
                e.g. all lap times get deleted. They come immediately after (1), and among them the
                finishing order is their original order in the PDF. (3) are DSQ drivers, whose
                entire quali. results are cancelled, e.g. rear wing technical infringement. They
                are the last and again among them the finishing order is the original order in the
                PDF. (4) are drivers whose results in *some* sessions are cancelled, e.g. 2025
                Bahrain Q2 Hulkenberg. They are placed between (2) and (3) and the order also
                follows the order in the PDF.
                """
                temp = temp.sort_values(by=['is_dsq', 'is_dnq', f'Q{q}', 'original_order'])
                temp['position'] = range(1, len(temp) + 1)
                temp['classification'] = temp.apply(
                    lambda x: SessionEntryImport(
                        object_type="SessionEntry",
                        foreign_keys=SessionEntryForeignKeys(
                            year=self.year,
                            round=self.round_no,
                            session=f'Q{q}' if self.session == 'quali' else f'SQ{q}',
                            car_number=x.NO
                        ),
                        objects=[
                            SessionEntryObject(
                                position=x.position,
                                is_classified=x.is_classified,
                                status=x.finishing_status
                            )
                        ]
                    ).model_dump(exclude_unset=True),
                    axis=1
                )
                data.extend(temp['classification'].tolist())
            return data

        df.to_json = to_json
        return df

    @staticmethod
    def _assign_session_to_lap(classification: pd.DataFrame, lap_times: pd.DataFrame) \
            -> pd.DataFrame:
        """TODO: probably need to refactor this later... To tedious now"""
        # TODO: this can be wrong. See #51
        # Assign session to lap No. in lap times, e.g. lap 8 is in Q2, using final classification
        classification = classification.copy()  # TODO: not the best practice?
        classification.Q1_LAPS = classification.Q1_LAPS.astype(float)
        classification.Q2_LAPS = classification.Q2_LAPS.astype(float) + classification.Q1_LAPS
        lap_times = lap_times.merge(classification[['NO', 'Q1_LAPS', 'Q2_LAPS']],
                                    left_on='car_no', right_on='NO', how='left')
        # TODO: should check if all merged. There shouldn't be any left only cars. Can have some
        #       right only cars, e.g. DNS, so all right only cars should be NOT CLASSIFIED drivers

        del lap_times['NO']
        lap_times['Q'] = 1
        lap_times.loc[lap_times.lap_no > lap_times.Q1_LAPS, 'Q'] = 2
        lap_times.loc[lap_times.lap_no > lap_times.Q2_LAPS, 'Q'] = 3
        # TODO: the lap immediately before the first Q2 and Q3 lap, i.e. the last lap in each
        #       session, should be a pit lap. Or is it? Crashed? Red flag?
        del lap_times['Q1_LAPS'], lap_times['Q2_LAPS']

        # Find which lap is the fastest lap, also using final classification
        """
        The final classification PDF identifies the fastest laps using calendar time, e.g.
        "18:17:46". In the lap times PDF, each driver's first lap time is the calendar time, e.g.
        "18:05:42"; for the rest laps, the time is the lap time, e.g. "1:24.160". Therefore, we can
        simply cumsum the lap times to get the calendar time of each lap, e.g.

        18:05:42 + 1:24.160 = 18:07:06.160

        The tricky part is rounding. Sometimes we have 18:17:15.674 -> 18:17:16, but in other times
        it is 18:17:46.783 -> 18:17:46. It seems to be not rounding to floor, not to ceil, and not
        to the nearest... Therefore, we allow one second difference. For a given driver, it's
        impossible to have two different laps finishing within one calendar second, so one second
        error in calendar time is ok to identify a lap.

        TODO: should check this against historical data
        """
        lap_times['calendar_time'] = lap_times.lap_time.apply(time_to_timedelta)
        lap_times.calendar_time = lap_times.groupby('car_no')['calendar_time'].cumsum()
        lap_times['is_fastest_lap'] = False
        for q in [1, 2, 3]:
            # Round to the floor
            # TODO: rewrite. What we need is Timedelta('0 days 16:07:13.470000') --> "16:07:13"
            lap_times['temp'] = lap_times.calendar_time.apply(
                lambda x: str(x).split('.')[0].split(' ')[-1]
            )
            lap_times = lap_times.merge(classification[['NO', f'Q{q}_TIME']],
                                        left_on=['car_no', 'temp'],
                                        right_on=['NO', f'Q{q}_TIME'],
                                        how='left')
            del lap_times['NO']
            # Plus one to the floor, i.e. allow one second error in the merge, and update the
            # previously non-matched cells using the new merge
            # TODO: rewrite as well. See above
            lap_times.temp = lap_times.calendar_time.apply(
                lambda x: str(x + pd.Timedelta(seconds=1)).split('.')[0].split(' ')[-1]
            )
            lap_times = lap_times.merge(classification[['NO', f'Q{q}_TIME']],
                                        left_on=['car_no', 'temp'],
                                        right_on=['NO', f'Q{q}_TIME'],
                                        how='left',
                                        suffixes=('', '_y'))
            del lap_times['NO'], lap_times['temp']
            lap_times = lap_times.fillna({f'Q{q}_TIME': lap_times[f'Q{q}_TIME_y']})
            del lap_times[f'Q{q}_TIME_y']

            # Check if all drivers in the final classification are merged
            temp = classification[['NO', f'Q{q}_TIME']].merge(
                lap_times[lap_times[f'Q{q}_TIME'].notna()][['car_no']],
                left_on='NO',
                right_on='car_no',
                indicator=True
            )
            temp = temp.dropna(subset=f'Q{q}_TIME')
            assert (temp['_merge'] == 'both').all(), \
                f"Some drivers' fastest laps in Q{q} cannot be found in lap times PDF: " \
                f"{', '.join([str(i) for i in temp[temp._merge != 'both']['NO']])}"
            lap_times.loc[lap_times[f'Q{q}_TIME'].notna(), 'is_fastest_lap'] = True
            del lap_times[f'Q{q}_TIME']
        return lap_times

    def _parse_lap_times(self) -> pd.DataFrame:
        """Parse "Qualifying/Sprint Quali./Shootout Session Lap Times" PDF"""
        doc = pymupdf.open(self.lap_times_file)
        dfs = []
        for page in doc:
            # Find "Lap Times"
            page = Page(page, file=self.lap_times_file)  # noqa: PLW2901
            quali_lap_times = page.search_for('Lap Times')
            if len(quali_lap_times) != 1:
                raise ParsingError(f'Find none or multiple "Lap Times" on p.{page.number} in '
                                   f'{self.lap_times_file}')
            b_lap_times = quali_lap_times[0].y1

            # Find the white strip immediately below "Lap Times", below which are the tables
            white_strip = page.search_for_white_strip(clip=(0, b_lap_times, page.w, page.h))
            if not white_strip:
                raise ParsingError(f'Expect at least a white strip below "Lap Times" on '
                                   f'p.{page.number} in {self.lap_times_file}. Found: '
                                   f'{white_strip}')
            t_all_drivers = white_strip[0]

            # Find all black horizontal lines (see RaceParser._parse_lap_analysis for details)
            black_lines = page.search_for_black_line(clip=(0, t_all_drivers, page.w, page.h),
                                                     min_length=0.25)
            if not black_lines:
                raise ParsingError(f'Could not find any black line below "Lap Times" on '
                                   f'p.{page.number} in {self.lap_times_file}')

            # Each line should be the separator between a table header and its body
            t_table_headers = []
            t_drivers = []
            b_tables = []
            for i in range(len(black_lines) - 1, -1, -1):
                # Table header is vertically between the black line and the white strip immediately
                # above the black line
                black_line = black_lines[i]
                white_strip = page.search_for_white_strip(clip=(0, 0, page.w, black_line))
                if not white_strip:
                    raise ParsingError(f'Could not find any white strips above the black line '
                                       f'at {black_line} on p.{page.number} in '
                                       f'{self.lap_times_file}. Found: {white_strip}')
                t_table_header = sorted(white_strip)[-1]
                header = page.get_text(clip=(0, t_table_header, page.w, black_line))

                # If no table header found, then it's the last black line at the bottom of the
                # page. Drop it
                if not ('NO' in header and 'TIME' in header):
                    black_lines.pop(i)
                    continue
                t_table_headers.insert(0, t_table_header)

                # The driver name is above the table header and the next white strip above
                if len(white_strip) < 2:  # noqa: PLR2004
                    raise ParsingError(f'Expect at least two white strips above the black line at '
                                       f'{black_line} on p.{page.number} in '
                                       f'{self.lap_times_file}. Found: {white_strip}')
                t_drivers.insert(0, white_strip[-2])

                # Table bottom is the next white strip below the black line
                white_strip = page.search_for_white_strip(clip=(0, black_line, page.w, page.h))
                if not white_strip:
                    raise ParsingError(f'Could not find any white strip below the black line at '
                                       f'{black_line} on p.{page.number} in {self.lap_times_file}')
                b_tables.insert(0, white_strip[0])

            # Two tables are vertically separated by a vertical white strip
            page.set_rotation(90)
            table_separators = page.search_for_white_strip(
                clip=(page.h - b_tables[-1], 0, page.h - t_drivers[0], page.w),
                height=0.03 * page.w  # White strip should occupy at least 3% of the page width
            )
            page.set_rotation(0)
            # A driver has at least one table, so at least two white strips: one to the left of the
            # table and the other to the right of it
            if len(table_separators) < 2:  # noqa: PLR2004
                raise ParsingError(f'Expect at least two vertical white strips below driver names '
                                   f'on p.{page.number} in {self.lap_times_file}. Found: '
                                   f'{table_separators}')

            # Loop through each table
            for i in range(len(t_drivers)):
                t_driver = t_drivers[i] + 1
                t_table_header = t_table_headers[i] + 1
                b_table_header = black_lines[i] - 1
                b_table = b_tables[i] + 1
                for j in range(0, len(table_separators) - 1):
                    l_table = max(0, table_separators[j] - 1)
                    r_table = table_separators[j + 1] + 1

                    # Get the driver name and car No.
                    driver = page.get_text(clip=(l_table, t_driver, r_table, t_table_header))
                    if not driver.strip():  # E.g., four tables on a page. The second row only has
                        continue            # one table, so we will have missing's here
                    car_no = re.match(r'^(\d+)\s+[A-Za-z ]+$', driver.strip())
                    if car_no:
                        car_no = int(car_no.group(1))
                    else:
                        raise ParsingError(f'Could not parse car No. in '
                                           f'({l_table:.2f}, {t_driver:.2f}, {r_table:.2f}, '
                                           f'{t_table_header:.2f}) on p.{page.number} in '
                                           f'{self.lap_times_file}: {driver}')

                    # Find the vertical white strip separating the two tables for the driver
                    page.set_rotation(90)
                    white_strip = page.search_for_white_strip(
                        clip=(page.h - b_table, l_table, page.h - t_table_header, r_table),
                        height=1  # Any height is fine
                    )
                    page.set_rotation(0)
                    # Table left, separator, and right. Three in total
                    if len(white_strip) != 3:  # noqa: PLR2004
                        raise ParsingError(f'Expected exactly three vertical white strips in '
                                           f'({l_table:.2f}, {t_table_header:.2f}, '
                                           f'{r_table:.2f}, {b_table:.2f}) on p.'
                                           f'{page.number} in {self.lap_times_file}. '
                                           f'Found: {white_strip}')
                    m_table = white_strip[1] + 1

                    # Parse each of the two tables of the driver
                    for l_tab, r_tab in [(l_table, m_table), (m_table, r_table)]:
                        # Refine table bottom
                        white_strip = page.search_for_white_strip(
                            clip=(l_tab, b_table_header, r_tab, page.h)
                        )
                        if not white_strip:
                            raise ParsingError(f'Could not find any white strip below the table in '
                                               f'({l_tab:.2f}, {b_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table:.2f}) on p.'
                                               f'{page.number} in {self.lap_times_file}')
                        b_table = white_strip[0]

                        # Get table header
                        cols = self._detect_cols(
                            page,
                            clip=(l_tab, t_table_header, r_tab, b_table_header),
                            col_min_gap=3,
                            min_black_line_length=0.5
                        )
                        if len(cols) != 2:  # noqa: PLR2004
                            raise ParsingError(f'Expected exactly two cols. in '
                                               f'({l_tab:.2f}, {t_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table_header:.2f}) on p.'
                                               f'{page.number} in {self.lap_times_file}. '
                                               f'Found: {cols}')
                        if cols[0].text != 'NO' or cols[1].text != 'TIME':
                            raise ParsingError(f'Expected "LAP" and "TIME" to be the two col. '
                                               f'names in ({l_tab:.2f}, {t_table_header:.2f}, '
                                               f'{r_tab:.2f}, {b_table_header:.2f}) on p.'
                                               f'{page.number} in {self.lap_times_file}. '
                                               f'Found: {cols}')
                        l_tab = cols[0].l - 1  # More accurate table left boundary  # noqa: PLW2901

                        # Vertical lines separating the two cols.
                        vlines = [l_tab, (cols[0].r + cols[1].l) / 2, r_tab]

                        # Horizontal lines are located by the grey and white rows
                        hlines = page.search_for_grey_white_rows(
                            clip=(l_tab, b_table_header + 1, r_tab, b_table + 1),
                            min_height=np.mean([i.b - i.t for i in cols]) - 1,
                            min_width=0.5
                        )

                        # Parse the table
                        df = self._parse_table_by_grid(page=page,
                                                       vlines=vlines,
                                                       hlines=hlines,
                                                       header_included=False)
                        if df.empty:  # E.g., DNS so no lap at all
                            continue
                        df.columns = [i.text.lower() for i in cols]
                        df = df.rename(columns={'no': 'lap_no', 'time': 'lap_time'})
                        df['lap_time_deleted'] = False

                        # Check if any crossed-out lap times
                        for k in range(len(hlines) - 1):
                            clip = (vlines[1], hlines[k], vlines[2], hlines[k + 1])
                            if page.has_horizontal_line(clip):
                                df.loc[k, 'lap_time_deleted'] = True

                        # Indicator for pit stop
                        df['pit'] = df.lap_no.str.contains('P', regex=False)
                        df.lap_no = df.lap_no.str.rstrip(' P').astype(int)
                        df['car_no'] = int(car_no)
                        dfs.append(df)

        # Clean up
        df = pd.concat(dfs, ignore_index=True)
        df.lap_no = df.lap_no.astype(int)
        df.car_no = df.car_no.astype(int)
        df = df.replace('', None)  # So empty cell will become NaN when casted to float
        df = self._assign_session_to_lap(self.classification_df, df)

        # Check if any fastest laps are wrong
        invalid_fastest_lap_drivers = set()
        def is_fastest_lap_valid() -> bool:
            """Check whether the fastest laps in lap times PDF match the ones in classification PDF

            This function checks, for each driver in each quali. session, whether his fastest lap
            time in lap times PDF is the same as the one in classification PDF. This is a necessary
            and sufficient condition to ensure that the fastest lap times are correct. However, it
            is necessary but not sufficient to guarantee that all lap times are correct/all laps
            are correctly matched to their quali. sessions.

            This partially fixes #51: when we get `False`here, there must be something wrong with
            linking laps to quali. sessions. In such case, we will have to use the fastest lap time
            from classification as fallback to ensure the fastest lap times are correct.
            """
            classification_df = self.classification_df[['NO', 'Q1', 'Q2', 'Q3']]
            lap_times_df = df[['car_no', 'lap_no', 'Q', 'lap_time', 'is_fastest_lap']]
            is_valid = True

            # Whether there is at most one fastest lap for each given driver in each given session
            """
            May have no fastest lap, e.g. a usual out lap, starting the flying lap, abort the lap,
            into pit. Two laps in total, but neither of them is a fastest lap. So here we check if
            #. of fastest laps per driver per session <= 1.
            """
            temp = (lap_times_df.groupby(['Q', 'car_no'])
                    .is_fastest_lap
                    .sum()
                    .reset_index(name='n_fastest_laps'))
            temp = temp[temp.n_fastest_laps > 1]
            if not temp.empty:
                is_valid = False
                invalid_fastest_lap_drivers.update(temp.car_no.unique())
                # TODO: should get a warning here

            # Compare the fastest lap times in lap times and classification PDFs
            lap_times_df = lap_times_df[
                lap_times_df.is_fastest_lap
                & (~lap_times_df.car_no.isin(invalid_fastest_lap_drivers))
            ]
            for q in [1, 2, 3]:
                temp = lap_times_df[lap_times_df.Q == q].merge(
                    classification_df,
                    left_on='car_no',
                    right_on='NO',
                    how='left',
                    validate='1:1'
                )
                temp = temp[temp.lap_time != temp[f'Q{q}']]
                if not temp.empty:
                    is_valid = False
                    invalid_fastest_lap_drivers.update(temp.car_no.unique())
                # TODO: should get a warning here
            return is_valid

        if not is_fastest_lap_valid():
            df = self._apply_fallback_fastest_laps(df, invalid_fastest_lap_drivers)

        # TODO: bad practice
        df.to_json = partial(quali_lap_times_to_json,
                             df=df, year=self.year, round_no=self.round_no, session=self.session)
        return df

    def _apply_fallback_fastest_laps(
            self,
            lap_times_df: pd.DataFrame,
            drivers_with_invalid_fastest_laps: set[int]
    ) -> pd.DataFrame:
        """
        Purge all lap times for drivers with invalid fastest laps and re-assign the fastest
        laps only.

        E.g., a driver has 5 laps in Q1, 6 laps in Q2, and does not make into Q3, and his Q1
        fastest lap is invalid, i.e. is not equal to the Q1 fastest lap time in classification
        PDF. We then drop all his 11 laps in both Q1 and Q2 from lap times df., and insert two
        new laps with the Q1 and Q2 fastest lap times from the classification PDF. The lap No.
        will be `None`. This fallback does discard all other laps, but this is intended, as the
        entire lap-session match is not reliable for this driver.
        """
        valid_laps = lap_times_df[~lap_times_df.car_no.isin(drivers_with_invalid_fastest_laps)]

        # Get fastest lap times from classification PDF for drivers with invalid fastest laps
        invalid_laps = []
        for car_no in drivers_with_invalid_fastest_laps:
            for q in [1, 2, 3]:
                fastest_lap = self.classification_df.loc[
                    self.classification_df.NO == car_no, f'Q{q}'
                ].to_numpy()[0]
                if pd.isna(fastest_lap) or (fastest_lap in ('DNF', 'DQ', 'DSQ', 'DNS')):
                    continue
                # Add a new lap with the fastest lap time
                invalid_laps.append({
                    'car_no': car_no,
                    'lap_no': None,  # `None` bc. we don't know which lap is the fastest lap
                    'pit': False,
                    'lap_time': fastest_lap,
                    'lap_time_deleted': False,
                    'Q': q,
                    'is_fastest_lap': True
                })

        invalid_laps = pd.DataFrame(invalid_laps)
        return pd.concat([valid_laps, invalid_laps], ignore_index=True)


class PitStopParser(BaseParser):
    def __init__(
            self,
            file: str | os.PathLike,
            year: int,
            round_no: int,
            session: RaceSessionT
    ):
        self.file = file
        self.year = year
        self.round_no = round_no
        self.session = session
        self._check_session()
        self.df = self._parse()

    def _check_session(self) -> None:
        if self.session not in get_args(RaceSessionT):
            raise ValueError(f'Invalid session: {self.session}. '
                             f'Valid sessions are {get_args(RaceSessionT)}')
        return

    def _parse(self) -> pd.DataFrame:
        doc = pymupdf.open(self.file)
        dfs = []
        # TODO: would be nice to add a test for page numbers: if more than one page, we should have
        #       "page x of xx" at the bottom right of each page
        for page in doc:  # Can have multiple pages, though usually only one. E.g., 2023 Dutch
            page = Page(page, file=self.file)  # noqa: PLW2901

            # Locate "Pit Stop Summary" title
            pit_stop_summary = page.search_for('Pit Stop Summary')
            if len(pit_stop_summary) != 1:
                raise ParsingError(f'Find none or multiple "Pit Stop Summary" on p.{page.number} '
                                   f'in {self.file}')
            b_title = pit_stop_summary[0].y1

            # Locate table header, vertically between the topmost black line and the white strip
            # immediately above the line
            black_line = page.search_for_black_line(clip=(0, b_title, page.w, page.h))
            if not black_line:
                raise ParsingError(f'Could not find a black horizontal line on p.{page.number} in '
                                   f'{self.file}')
            b_table_header = black_line[0] - 1
            t_table_body = black_line[0] + 1
            white_strip = page.search_for_white_strip(clip=(0, 0, page.w, b_table_header))
            if not white_strip:
                raise ParsingError(f'Could not find a white horizontal strip above the black line '
                                   f'on p.{page.number} in {self.file}')
            t_table_header = white_strip[-1] + 1

            # Get col. names
            cols = self._detect_cols(page,
                                     clip=(0, t_table_header, page.w, b_table_header),
                                     col_min_gap=2)  # Very wide cols., so allow larger gaps
            if [i.text for i in cols] != ['NO', 'DRIVER', 'ENTRANT', 'LAP', 'TIME OF DAY', 'STOP',
                                          'DURATION', 'TOTAL TIME']:
                raise ParsingError(f'Table cols. are not as expected on p.{page.number} in '
                                   f'{self.file}. Found: {cols}')

            # Table bottom is the first white strip below the table header
            white_strip = page.search_for_white_strip(clip=(0, t_table_body, page.w, page.h))
            if not white_strip:
                raise ParsingError(f'Could not find a white horizontal strip below the table '
                                   f'header on p.{page.number} in {self.file}')
            b_table = white_strip[0] + 1

            # Locate the horizontal positions of each col.
            col_pos = self._detect_cols(page,
                                        clip=(0, t_table_header, page.w, b_table),
                                        col_min_gap=2)
            if len(cols) != len(col_pos):
                raise ParsingError(f'Number of detected cols. does not match number of col. names '
                                   f'on p.{page.number} in {self.file}: {cols} vs. {col_pos}')
            vlines = ([col_pos[0].l - 1]
                      + [(col_pos[i].r + col_pos[i + 1].l) / 2 for i in range(len(col_pos) - 1)]
                      + [col_pos[-1].r + 1])

            # Get row positions by white and grey rectangles
            hlines = page.search_for_grey_white_rows(
                clip=(vlines[0], t_table_body, vlines[-1], b_table),
                min_height=np.mean([i.b - i.t for i in cols])
            )

            # Parse
            df = self._parse_table_by_grid(
                page=page,
                vlines=vlines,
                hlines=hlines,
                header_included=False
            )
            df.columns = [i.text for i in cols]
            dfs.append(df)

        # Clean up the table
        df = pd.concat(dfs, ignore_index=True)
        del dfs
        df = df[['NO', 'LAP', 'TIME OF DAY', 'STOP', 'DURATION']].reset_index(drop=True)
        df = df.rename(columns={
            'NO': 'car_no',
            'LAP': 'lap',
            'TIME OF DAY': 'local_time',
            'STOP': 'stop_no',
            'DURATION': 'duration'
        })
        df.car_no = df.car_no.astype(int)
        df.lap = df.lap.astype(int)
        df.stop_no = df.stop_no.astype(int)

        def to_json() -> list[dict]:
            pit_stop = df.copy()
            pit_stop['pit_stop'] = pit_stop.apply(
                lambda x: PitStopObject(
                    number=x.stop_no,
                    duration=duration_to_millisecond(x.duration),
                    local_timestamp=x.local_time
                ),
                axis=1
            )
            pit_stop['entry'] = pit_stop.apply(
                lambda x: PitStopForeignKeys(
                    year=self.year,
                    round=self.round_no,
                    session=self.session if self.session == 'race' else 'SR',
                    car_number=x.car_no,
                    lap=x.lap
                ), axis=1
            )
            return pit_stop.apply(
                lambda x: PitStopData(
                    object_type="PitStop",
                    foreign_keys=x.entry,
                    objects=[x.pit_stop]
                ).model_dump(exclude_unset=True),
                axis=1
            ).tolist()

        # TODO: bad practice
        df.to_json = to_json
        return df


if __name__ == '__main__':
    pass
