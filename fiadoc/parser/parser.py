# -*- coding: utf-8 -*-
import os
import re
import warnings
from functools import cached_property, partial
from typing import Any, Literal, Optional, get_args

import numpy as np
import numpy.typing as npt
import pandas as pd
import pymupdf
from scipy.ndimage import find_objects, label

from .._constants import DPI, QUALI_DRIVERS
from ..drivers import Drivers
from ..models.classification import SessionEntryImport, SessionEntryObject
from ..models.driver import (
    DriverImport,
    DriverObject,
    RoundEntryImport,
    RoundEntryObject,
    TeamDriverImport,
    TeamDriverObject
)
from ..models.foreign_key import (
    PitStopForeignKeys,
    RoundEntryForeignKeys,
    SessionEntryForeignKeys,
    TeamDriverForeignKeys
)
from ..models.lap import LapImport, LapObject
from ..models.pit_stop import PitStopData, PitStopObject
from ..utils import duration_to_millisecond, time_to_timedelta
from .page import BBox, Page, ParsingError, TextBlock

PracticeSessionT = Literal['fp', 'fp1', 'fp2', 'fp3']
RaceSessionT = Literal['race', 'sprint']
QualiSessionT = Literal['quali', 'sprint_quali']

DRIVERS = Drivers(cache_dir=os.environ.get('FIADOC_CACHE_DIR', None))

WHITE_STRIP_MIN_HEIGHT = 10  # A table should end with a white strip with at least 10px height
LINE_MIN_VGAP = 5  # If two horizontal lines are vertically separated by less than 5px, they are
                   # considered to be the same line


class BaseParser:
    """Base class for all parsers

    Provides some common functionality, such as locating the title, getting col. positions, etc.
    Not meant to be instantiated directly, but rather to be inherited by others.
    """
    @staticmethod
    def _detect_cols(
            page: Page,
            clip: BBox,
            col_min_gap: float = 1.1,
            min_black_line_length: float = 0.9
    ) -> list[TextBlock]:
        """
        Search for table header/cols. in the `clip` area

        The high-level idea is:

        1. convert the `clip` area into an image/pixel map
        2. for each character, mask it into black rectangle using `scipy.ndimage.label`. This is
           possible because all black pixels are text (sufficient and necessary). We will address
           the black table lines below
        3. merge multiple black rectangles into a single one if they are sufficiently close to
           each other, i.e. group letters into words
        4. get the text inside each block

        :param col_min_gap: Minimum vertical gap between two cols., relative to width of the median
                            thinnest char. Default is 1.1, i.e. 110% of the width of the median
                            thinnest char. E.g., we have "D", "r", "i", "v", "e", "r", some space,
                            "N", "a", "t", some space, "T", "e", "a", "m". We'd like to group them
                            into "Driver", "Nat", and "Team", etc. So for chars. vertically closer
                            than "some space", they should be grouped together. This `col_min_gap`
                            is basically how big is "some space", relative to the median thinnest
                            char. The reason to use median is because of extremely thin chars.,
                            like "." in "No.". Using the width of "." may break "Nat" into "N",
                            "a", and "t". From thin to fat, we often have ".", "I", "a", "W".
                            Usually the width of "a" is a good reference: chars. within a distance
                            of width of "a" are in considered to be in a same word, while chars.
                            farther from this width will be considered to be in two different words
        :param min_black_line_length: Minimum length of a black line, relative to the corresponding
                                      edge length of the `clip` area. Default is 0.9. That is, any
                                      black vertical lines longer than 90% of the height of `clip`,
                                      and any black horizontal lines longer than 90% of the width
                                      of `clip`, are ignored
        :return: List of TextBlock representing the detected cols.
        """
        # Get the pixmap of `clip` area
        # TODO: create a get pixmap method in Page
        pixmap: pymupdf.Pixmap = page.get_pixmap(clip=clip, dpi=DPI)
        arr: npt.NDArray[np.uint8] = (np.frombuffer(buffer=pixmap.samples_mv, dtype=np.uint8)
                                      .reshape((pixmap.height, pixmap.width, 3)))

        # Replace rows and cols. w/ almost all black pixels with white pixels (those are lines not
        # text). After this step, all black pixels should be texts only
        """
        There are usually only three colours in the PDF: black (RGB ~= 21), white (0), and grey
        (~184 or ~232), so use RGB = 128 as a cutoff for "black"
        """
        arr[np.mean(arr < 128, axis=(1, 2)) >= min_black_line_length] = 255  # noqa: PLR2004
        arr[:, np.mean(arr < 128, axis=(0, 2)) >= min_black_line_length] = 255  # noqa: PLR2004

        # Mask text with rectangles, i.e. label connected components
        labelled, n_features = label(arr < 128)  # noqa: PLR2004
        if n_features == 0:
            return []
        slices: list[tuple[slice, slice, slice]] = sorted(find_objects(labelled),
                                                          key=lambda x: x[1].start)

        # For debug: visualise the detected slices
        # temp = arr.copy()
        # for s in slices:
        #     temp[s[0], s[1]] = [0, 0, 0]

        # Merge the rectangles/slices that are close to each other
        """
        "Close" is defined as smaller than the width of the median thinnest char., which is often
        "a". The gap between two cols. are usually very wide, except "NAT" and "ENTRANT", between
        which the gap is roughly a normal whitespace. If we pick a too wide "close", "NAT" and
        "ENTRANT" will be recognised as a single text block "NAT ENTRANT". If "close" is too
        narrow, then "NAT" may be broken into "N", "A", and "T". Generally speaking, using the
        median thinnest char.'s width (w/ ~10% buffer) satisfies both conditions.
        """
        med_char_width: float = np.median([s[1].stop - s[1].start for s in slices]) * col_min_gap
        merged_slices: list[tuple[slice, slice, slice]] = [slices[0]]
        for s in slices[1:]:
            # If the current char. is sufficiently close to the last word, then it belongs to the
            # last word
            if s[1].start - merged_slices[-1][1].stop < med_char_width:
                last = merged_slices[-1]
                merged_slices[-1] = (
                    slice(min(last[0].start, s[0].start), max(last[0].stop, s[0].stop)),
                    slice(min(last[1].start, s[1].start), max(last[1].stop, s[1].stop)),
                    last[2]
                )
            else:
                # If far from the last word, then the current char. is the start of a new word
                merged_slices.append(s)

        # For debug: visualise the detected words/merged slices
        # temp = arr.copy()
        # for s in merged_slices:
        #     temp[s[0], s[1]] = [0, 0, 0]

        # Get the text of these words
        cols: list[TextBlock] = []
        for s in merged_slices:
            l, t, r, b = page._transform_bbox(bbox=(s[1].start, s[0].start, s[1].stop, s[0].stop),
                                              from_page_bound=(0, 0, arr.shape[1], arr.shape[0]),
                                              to_page_bound=clip)
            l -= 1  # Give 1px buffer
            t -= 1
            r += 1
            b += 1
            texts = page.get_text('text', clip=(l, t, r, b), small_area=True)
            if len(texts) != 1:
                raise ParsingError(f'Expected one text block for col. name. Found {texts} inside '
                                   f'({clip[0]:.2f}, {clip[1]:.2f}, {clip[2]:.2f}, {clip[3]:.2f}) '
                                   f'on p.{page.number} in {page.file}')
            text = texts[0]
            if text.text:
                # The "/" in "KM/H" is often mis-OCRed. Manually correct it here
                if re.match(r'KM\SH', text.text):
                    text.text = 'KM/H'
                text.bbox = (l, t, r, b)
                cols.append(text)
            else:
                raise ParsingError(f'Unable to get text in the table header inside '
                                   f'({l:.2f}, {t:.2f}, {r:.2f}, {b:.2f}) on p.{page.number} in '
                                   f'{page.file}')
        return cols

    @staticmethod
    def _normalise_textblock(tbs: Any, merge_multi_tbs: bool = False) -> Optional[str | int] | Any:
        """Convert a single or a list of TextBlock into a string

        If it's not a TextBlock or a list of TextBlocks, return it directly. Otherwise:

        1. It's a single textblock, or in the list there is only one textblock, no matter it's
           normal or superscript or whatever, return its text directly
        2. If multiple textblocks are there, and only one is normal text, i.e. no strikethrough or
           superscript, return that normal one's text
        3. if multiple normal textblocks, depending on `merge_multi_tbs`, either merge their text
           together using a white space, or raise an error

        If `tbs` is None, and will return `None`. If the final returned text is pure digits, will
        convert it to `int`

        TODO: should we put this method here? Or in some other class?
        """
        if tbs is None:
            return None

        # 1.
        if isinstance(tbs, TextBlock):
            if not tbs.text:
                return None
            return int(tbs.text) if tbs.text.isdigit() else tbs.text

        # 0.
        if (not isinstance(tbs, list)) or any(not isinstance(tb, TextBlock) for tb in tbs):
            return tbs

        # 1.
        if len(tbs) == 1:
            if not tbs[0].text:
                return None
            return int(tbs[0].text) if tbs[0].text.isdigit() else tbs[0].text

        # 2.
        normal_texts = [tb.text
                        for tb in tbs
                        if (not tb.superscript) and tb.text]
        if not normal_texts:
            return None
        if len(normal_texts) == 1:
            return int(normal_texts[0]) if normal_texts[0].isdigit() else normal_texts[0]

        # 3.
        if merge_multi_tbs:
            return ' '.join([tb.text for tb in tbs if tb.text])
        raise ParsingError(f'Expected only one normal textblock. Found multiple: {tbs}')


class EntryListParser(BaseParser):
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

    def _parse(self) -> pd.DataFrame:
        """
        :return: Df. with cols. of ["car_no", "driver", "nat", "team", "constructor"]
        """
        # Go to the page with "No.", "Driver", "Nat", "Team", and "Constructor". These col. names
        # should appear in the top half of the page
        doc = pymupdf.open(self.file)
        found = False
        page: Page
        for page in doc:
            page = Page(page, file=self.file)  # noqa: PLW2901
            tb = page.get_text('text', clip=(0, 0, page.w, page.h * 0.5))
            if all(i in tb[0].text for i in ['No.', 'Driver', 'Nat', 'Team', 'Constructor']):
                found = True
                break
        if not found:
            doc.close()
            raise ValueError(f'Could not find any page containing entry list table in {self.file}')

        # The table starts under the bottommost horizontal black line
        if black_lines := page.search_for_black_lines(clip=(0, 0, page.w, page.h * 0.5),
                                                      min_length=0.5):
            t_table_header = black_lines[-1] + 1  # Add 1px buffer
        else:
            doc.close()
            raise ParsingError(f'Cannot find the black line above the table header in {self.file}')

        # Table header is below the black line and above a white strip
        if white_strips := page.search_for_white_strips(clip=(0, t_table_header, page.w, page.h),
                                                        height=0.5):  # Very short white strip
            if len(white_strips) < 2:  # noqa: PLR2004
                doc.close()
                raise ParsingError(f'Expected at least two white strips below the bottommost '
                                   f'black line in {self.file}. Found: {white_strips}')
            b_table_header = white_strips[1] + 1  # 1px buffer
        else:
            doc.close()
            raise ParsingError(f'find no white strip below the table header in {self.file}')

        # Col. positions
        cols = self._detect_cols(page,
                                 clip=(0, t_table_header, page.w, b_table_header),
                                 col_min_gap=1.5,
                                 min_black_line_length=0.5)
        if [i.text.lower() for i in cols] != ['no.', 'driver', 'nat', 'team', 'constructor']:
            raise ParsingError(f'Expected cols. "No.", "Driver", "Nat", "Team", and "Constructor" '
                               f'in {self.file}. Got: {cols}')
        vlines = [cols[0].bbox[0] - 1,
                  cols[1].bbox[0] - 1,
                  cols[2].bbox[0] - 1,
                  (cols[2].bbox[2] + cols[3].bbox[0]) / 2,
                  cols[4].bbox[0] - 1,
                  page.w]  # Vertical lines separating the cols.
        col_row_height = np.mean([i.bbox[3] - i.bbox[1] for i in cols])  # Table header row height

        # Table ends above a sufficiently tall white strip
        if white_strips := page.search_for_white_strips(clip=(0, b_table_header, page.w, page.h),
                                                        height=col_row_height):
            b_table = white_strips[0] + 1
        else:
            doc.close()
            raise ParsingError(f'Cannot find the white strip below the table in {self.file}')

        # Row positions, identified by very short white strips between two consecutive rows
        white_strips = page.search_for_white_strips(clip=(0, b_table_header, page.w, b_table),
                                                    height=0.1)
        if not white_strips:
            raise ParsingError(f'Cannot find any white strip separating rows in the table in '
                               f'{self.file}')
        """
        We want to know the vertical gap between two rows, so we can pass it to `tol` in
        `.parse_table_by_grid()`. The white strips found above are their top positions. So between
        two consecutive white strips, there is a row plus a vertical gap to the next row.
        Therefore, vertical gap is approx. the dist. between two consecutive white strips minus the
        row height, which is already found above as `col_row_height`.
        """
        row_gap = np.mean(np.diff(white_strips)) - col_row_height
        # Exclude, if any, the topmost and bottommost white strips, if they are the white spaces
        # above and below the table
        hlines = [i + row_gap / 2 for i in white_strips[1:-1]]
        # If the top white strip is sufficiently far from the table header, then it is the white
        # space between row 0 and 1. Otherwise, it's the separator between the table header and row
        # 0, which does not need to be included
        if abs(white_strips[0] - b_table_header) > col_row_height / 2:
            hlines.insert(0, white_strips[0] - row_gap / 2)
        # Similarly for the bottom white strip
        if abs(b_table - white_strips[-1]) > col_row_height / 2:
            hlines.append(white_strips[-1] + row_gap / 2)
        hlines.insert(0, b_table_header)
        hlines.append(b_table + row_gap / 2) # Now `hlines` are horizontal lines separating rows

        # Parse the table using the grid above
        df = page.parse_table_by_grid(vlines=vlines,
                                      hlines=hlines,
                                      allow_multiple_texts_per_cell=[0],  # Allow superscripts
                                      header_included=False,
                                      tol=row_gap)
        df.columns = ['car_no', 'driver', 'nat', 'team', 'constructor']

        def identify_reserve(tbs: list[TextBlock]) -> Optional[int]:
            if len(tbs) == 1:
                return None
            if len(tbs) >= 3 or len(tbs) == 0:  # noqa: PLR2004
                raise ParsingError(f'Find none or more than two text blocks in "No." col. in '
                                   f'{self.file}: {tbs}')
            # Two text blocks. One is the car No. and the other, which is a superscript, indicates
            # reserve driver
            normal_driver: Optional[str] = None
            reserve_driver: Optional[str] = None
            for tb in tbs:
                if tb.superscript:
                    reserve_driver = tb.text
                else:
                    normal_driver = tb.text
            if (not normal_driver) or (not reserve_driver):
                raise ParsingError(f'Cannot identify normal or reserve driver in "No." col. in '
                                   f'{self.file}: {tbs}')
            return int(reserve_driver)

        df['has_reserve'] = df.car_no.apply(identify_reserve)

        # Check if we have a second table for reserve drivers, by searching for numbers in the No.
        # col. below the table
        reserve_driver_nos = [
            i for i in page.get_text('words',
                                     clip=(vlines[0], b_table + 1, vlines[1], page.h))
            if i.text.isdigit()
        ]

        # Check consistency between the reserve drivers indicated in the main table and the reserve
        # drivers we find above (#23)
        """
        We have two cases:

        1. both tables have reserve drivers
        2. the main table has some reserve drivers, but we find no reserve drivers here
        3. the main table has no reserve drivers, but we do find some here

        1. is what should happen. 2. either means the FIA doc. is wrong (see #23), or our parser
        mis-identify something. Raise a warning here. 3. means the main table missed some reserve
        drivers. This is an error for sure.
        """
        has_reserves_in_main_table = df.has_reserve.notna().any()
        reserves_found = (len(reserve_driver_nos) > 0)
        if has_reserves_in_main_table == reserves_found:  # All good
            pass
        elif has_reserves_in_main_table and (not reserves_found):
            warnings.warn(f'Found reserve drivers in the main table, but no reserve driver table '
                          f'in {self.file}. Either the FIA doc. is wrong (see #23), or our parser '
                          f'mis-identified something')
        else:
            doc.close()
            raise ParsingError(f'Reserve drivers found in the main table does not agree with'
                               f'those below the main table in {self.file}')

        # Parse the reserve driver table, if any
        if reserves_found:
            # Get the top and bottom of the reserve driver table
            """
            We can't rely on the bboxes of `reserve_driver_nos` to identify the rows, or the table
            position, because PyMuPDF returns very weird bbox if a text has superscript. Therefore,
            we still locate the table using white strips.

            The table top will be the last white strip between the main table bottom and the
            bottommost reserve driver No. We use the bottommost to make sure, even if the bbox from
            PyMuPDF is wrong, we are conservative enough to include at least the zero-th row.

            Table bottom is identified similarly: the first tall white strip below the bottommost
            reserve driver No.
            """
            if white_strips := page.search_for_white_strips(
                    clip=(0, b_table, page.w, max(i.bbox[3] for i in reserve_driver_nos)),
                    height=col_row_height
            ):
                t_reserve_table = white_strips[-1] + 1
            else:
                doc.close()
                raise ParsingError(f'Cannot find the white strip above the reserve driver table '
                                   f'in {self.file}')
            if white_strips := page.search_for_white_strips(
                    clip=(0, max(i.bbox[3] for i in reserve_driver_nos), page.w, page.h),
                    height=col_row_height
            ):
                b_reserve_table = white_strips[0] + 1
            else:
                doc.close()
                raise ParsingError(f'Cannot find the white strip below the reserve driver table '
                                   f'in {self.file}')

            # Identify rows
            white_strips = page.search_for_white_strips(
                clip=(0, t_reserve_table, page.w, b_reserve_table),
                height=0.1
            )
            hlines_reserve = [i + col_row_height * 0.1 for i in white_strips[1:-1]]
            if abs(white_strips[0] - t_reserve_table) > col_row_height / 2:
                hlines_reserve.insert(0, white_strips[0])
            if abs(b_reserve_table - white_strips[-1]) > col_row_height / 2:
                hlines_reserve.append(white_strips[-1])
            hlines_reserve.insert(0, t_reserve_table)
            hlines_reserve.append(b_reserve_table)
            # TODO: add row gap to hlines_reserve, like what we do for the main table?

            # Parse the reserve driver table
            reserves_df = page.parse_table_by_grid(vlines=vlines,
                                                   hlines=hlines_reserve,
                                                   allow_multiple_texts_per_cell=[0],
                                                   header_included=False,
                                                   tol=col_row_height * 0.3)
            reserves_df.columns = ['car_no', 'driver', 'nat', 'team', 'constructor']
            reserves_df['reserve_for'] = reserves_df.car_no.apply(identify_reserve)
            df = pd.concat([df, reserves_df], ignore_index=True)

        df = df.map(self._normalise_textblock)

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
                    driver_id = DRIVERS.get_driver_id(year=self.year, full_name=x.driver)
                    for warn in w:
                        if 'Creating a new driver ID' in str(warn.message):
                            new_driver_objects.append(
                                DriverObject(
                                    reference=driver_id,
                                    forename=DRIVERS.get_first_name(x.driver),
                                    surname=DRIVERS.get_last_name(x.driver),
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
        # TODO: do we need this? Isn't pydantic doing this already?
        if self.session not in get_args(PracticeSessionT):
            raise ValueError(f'Invalid session: {self.session}. Valid sessions are: '
                             f'{get_args(PracticeSessionT)}')
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
            warnings.warn('Lap times PDF is missing. Can get fastest laps ONLY from the '
                          'classification PDF')
            return self._apply_fallback_fastest_laps()

    def _parse_classification(self) -> pd.DataFrame:
        """Parse "(First/Second/Third) Practice Final Classification" PDF

        The output dataframe has columns [driver No., laps completed, total time,
        finishing position, finishing status, fastest lap time, fastest lap speed, fastest lap No.]
        """
        # Find the page with "Practice Session Classification", on which the table is located
        doc = pymupdf.open(self.classification_file)
        page: Page
        classification: Optional[list[TextBlock]] = None
        for page in doc:
            page = Page(page, file=self.classification_file)  # noqa: PLW2901
            if classification := page.search_for('Practice Session Classification'):
                if len(classification) > 1:
                    doc.close()
                    raise ParsingError(f'Found more than one "Practice Session Classification" on '
                                       f'p.{page.number} in {self.classification_file}')
                break
        if not classification:
            doc.close()
            raise ParsingError(f'"Practice Session Classification" not found on any page in '
                               f'{self.classification_file}')
        page_no_str = f'p.{page.number} in {page.file}'  # For error/warning messages

        # Position of "Practice Session Classification", below which is the table
        b_classification = classification[0].b

        # Find the first black horizontal line below "Practice Session Classification"
        if black_lines := page.search_for_black_lines(clip=(0, b_classification, page.w, page.h)):
            t_table_body = black_lines[0]  # Topmost black line below "Final Classification"
        else:
            doc.close()
            raise ParsingError(f'Cannot find the black line separating table header and table '
                               f'body below "Practice Session Classification" on {page_no_str}')

        # Get cols.
        cols = self._detect_cols(page,
                                 clip=(0, b_classification + 1, page.w, t_table_body - 1),
                                 col_min_gap=1)
        if not cols:
            raise ParsingError(f'Could not locate cols. in the table header on {page_no_str}')
        if [i.text.upper() for i in cols] != ['NO', 'DRIVER', 'NAT', 'ENTRANT', 'TIME', 'LAPS',
                                              'GAP', 'INT', 'KM/H', 'TIME OF DAY']:
            doc.close()
            raise ParsingError(
                f'Got unexpected or miss some cols. on {page_no_str}. Expected "NO", "DRIVER", '
                f'"NAT", "ENTRANT", "TIME", "LAPS", "GAP", "INT", "KM/H", and "TIME OF DAY". Got: '
                f'{[i.text for i in cols]}'
            )
        vlines = [0,
                  cols[0].bbox[0] - 1,
                  (cols[0].bbox[2] + cols[1].bbox[0]) / 2,
                  cols[2].bbox[0] - 1,
                  (cols[2].bbox[2] + cols[3].bbox[0]) / 2,
                  1.5 * cols[4].l - 0.5 * cols[4].r,  # Left of "TIME" - half-width of "TIME"
                  cols[5].bbox[0],
                  cols[5].bbox[2],
                  (cols[6].bbox[2] + cols[7].bbox[0]) / 2,
                  (cols[7].bbox[2] + cols[8].bbox[0]) / 2,
                  cols[9].bbox[0],
                  cols[9].bbox[2]]
        col_row_height = np.mean([i.bbox[3] - i.bbox[1] for i in cols])

        # The first white strip below the table header, which is the bottom of the table
        if white_strips := page.search_for_white_strips(clip=(0, t_table_body, page.w, page.h),
                                                        height=col_row_height):
            b_table = sorted(white_strips)[0]
        else:
            doc.close()
            raise ParsingError(f'Could not find table bottom by white strip on p.{page.number} in '
                               f'{self.classification_file}')

        # Horizontal lines separating the rows
        hlines = page.search_for_grey_white_rows(clip=(0, t_table_body + 1, page.w, b_table + 1),
                                                 min_height=col_row_height / 2)

        # Parse the table using the grid above
        df = page.parse_table_by_grid(vlines=vlines, hlines=hlines, header_included=False)
        df = df.map(self._normalise_textblock)
        df.columns = ['finishing_position', 'car_no', 'driver', 'nat', 'team', 'fastest_lap_time',
                      'laps_completed', 'gap', 'int', 'avg_speed', 'fastest_lap_calender_time']
        df.fastest_lap_time = df.fastest_lap_time.apply(duration_to_millisecond)

        """
        We don't really know the finishing status of the driver from the table. E.g., in 2024
        Australian FP1, Albon crashed, but before that, he already set several valid laps, so his
        name is in the PDF, without any mark about the crash. So we set the finishing status to be
        missing for everyone, because we can't infer that from the PDF.

        And all drivers in the table will be classified, because as long as they make a lap that
        counts, they are classified and in the PDF. DNS or other not classified drivers won't be in
        the PDF in the first place.

        TODO: should we add back not classified drivers manually?
        """

        def to_json() -> list[dict]:
            return df.apply(
                lambda x: SessionEntryImport(
                    object_type='SessionEntry',
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
        page: Page
        for page in doc:
            # Find "Lap Times"
            page = Page(page, file=self.lap_times_file)  # noqa: PLW2901
            page_no_str = f'p.{page.number} in {page.file}'
            lap_times = page.search_for('Lap Times')
            if len(lap_times) != 1:
                doc.close()
                raise ParsingError(f'Find none or multiple "Lap Times" on {page_no_str}')
            b_lap_times = lap_times[0].y1
            # Will use height of "Lap Times" as a reference for the white strips between drivers
            lap_times_height = lap_times[0].bbox[3] - lap_times[0].bbox[1]

            # Find the white strip immediately below "Lap Times", below which are the tables
            if white_strips := page.search_for_white_strips(clip=(0, b_lap_times, page.w, page.h)):
                t_all_drivers = white_strips[0]
            else:
                doc.close()
                raise ParsingError(f'Expect at least a white strip below "Lap Times" on '
                                   f'{page_no_str}. Found: {white_strips}')

            # Find all black horizontal lines (see RaceParser._parse_lap_analysis() for details)
            """
            One page has six tables side by side at most, and one table has one black line below
            its table header, so on average one black line occupies at most 1/6 of the page width.
            In real PDF, that 1/6 is even smaller, because there are of course some white spaces
            between tables. Therefore, we set the `min_length` to be 1/8 roughly.
            """
            black_lines = page.search_for_black_lines(clip=(0, t_all_drivers, page.w, page.h),
                                                      min_length=0.125)
            if not black_lines:
                doc.close()
                raise ParsingError(f'Could not find any black line below "Lap Times" on '
                                   f'{page_no_str}')

            # Exclude the bottommost black line, which is the footnote separator, not a table
            bottom_black_line: Optional[float] = None
            if long_black_lines := page.search_for_black_lines(
                    clip=(0, black_lines[0], page.w, page.h),
                    min_length=0.7
            ):
                black_lines = [l for l in black_lines
                               if not any(np.isclose(l, long_black_line, atol=5)
                               for long_black_line in long_black_lines)]
                bottom_black_line = long_black_lines[0]
            if not black_lines:
                doc.close()
                raise ParsingError(f'Could not find any black line below "Lap Times" on '
                                   f'{page_no_str}')

            # Find vertical white spaces separating the three side-by-side drivers. These white
            # strips should be relatively tall. We use half of the "Lap Times" height as the
            # threshold here
            if len(black_lines) == 1:  # If only one row of drivers on the page, then the bottom of
                b_page_content = page.h
                if bottom_black_line:  # the search area is page bottom, excl. footnote
                    b_page_content = bottom_black_line - 1
                    if page_no_text := page.search_for(
                            f'Page {page.number + 1} of',
                            clip=(0, black_lines[-1], page.w, b_page_content)
                    ):
                        b_page_content = min(b_page_content, page_no_text[0].y0 - 1)
                clip = (page.h - b_page_content, 0, page.h - black_lines[0] - 1, page.w)
            else:
                clip = (page.h - black_lines[-1] + 1, 0, page.h - black_lines[0] - 1, page.w)
            page.set_rotation(90)
            driver_separators = page.search_for_white_strips(clip=clip,
                                                             height=lap_times_height * 0.5)
            page.set_rotation(0)
            # A driver has at least one table, so at least two white strips: one to the left of the
            # table and the other to the right of it
            if len(driver_separators) < 2:  # noqa: PLR2004
                doc.close()
                raise ParsingError(f'Expect at least two vertical white strips separating drivers '
                                   f'on {page_no_str}. Found: {driver_separators}')
            elif len(driver_separators) > 4:  # noqa: PLR2004
                doc.close()
                raise ParsingError(f'Expect at most four white strips separating drivers on '
                                   f'{page_no_str}. Found: {driver_separators}')

            # Each line should be the separator between a table header and its body, so use these
            # black lines to locate the tables
            for black_line in black_lines:
                # Table header is vertically between the first and second white strips above the
                # black line
                white_strips = page.search_for_white_strips(clip=(0, 0, page.w, black_line),
                                                            height=lap_times_height / 3)
                if len(white_strips) < 2:  # noqa: PLR2004
                    doc.close()
                    raise ParsingError(f'Find one or no white strip above the black line at '
                                       f'{black_line} on {page_no_str}: {white_strips}. Expected '
                                       f'at least two')
                t_driver = white_strips[-2] + 1
                t_table_header = white_strips[-1] + 1

                # Bottom of the table header is the very thin white strip above the black line
                if white_strip := page.search_for_white_strips(
                        clip=(0, t_table_header, page.w, black_line),
                        height=1
                ):
                    b_table_header = white_strip[-1] + 1
                else:
                    doc.close()
                    raise ParsingError(f'Could not find any white strip separating table header'
                                       f'and body around {black_line} on {page_no_str}')

                # The shared bottom of all tables in the row is the white strip below black line
                if white_strips := page.search_for_white_strips(
                        clip=(0, black_line, page.w, page.h)
                ):
                    b_tables = white_strips[0] + 1
                else:
                    doc.close()
                    raise ParsingError(f'Could not find any white strip below the black line at '
                                       f'{black_line} on {page_no_str}')

                # Parse each of the three side by side drivers' tables
                for l_driver, r_driver in zip(driver_separators[:-1], driver_separators[1:]):
                    l_driver += 1  # noqa: PLW2901
                    r_driver += 1  # noqa: PLW2901

                    # Get the driver name and car No.
                    driver = page.get_text(clip=(l_driver, t_driver, r_driver, t_table_header))
                    # Skip if no driver. E.g., four tables on a page. The second row only has one
                    # table, so should skip two in the second row
                    if (not driver) or (not ''.join(i.text for i in driver)):
                        continue
                    if match := re.match(r'^(\d+)\s+[A-Za-z ]+$', driver[0].text):
                        car_no = int(match.group(1))
                    else:
                        doc.close()
                        raise ParsingError(f'Could not parse car No. in '
                                           f'({l_driver:.2f}, {t_driver:.2f}, {t_driver:.2f}, '
                                           f'{t_table_header:.2f}) on {page_no_str}: {driver}')

                    # Find the thin vertical white strip separating two tables of the driver
                    page.set_rotation(90)
                    table_separators = page.search_for_white_strips(
                        clip=(page.h - b_tables, l_driver, page.h - t_table_header, r_driver),
                        height=1
                    )
                    page.set_rotation(0)
                    if np.isclose(table_separators[0], l_driver, atol=5):   # Don't need driver
                        table_separators.pop(0)                             # separators here. Will
                    if np.isclose(table_separators[-1], r_driver, atol=5):  # add back manually
                        table_separators.pop(-1)                            # later
                    if table_separators:
                        table_separators.insert(0, l_driver)
                        table_separators.append(r_driver)
                    else:
                        table_separators = [l_driver, r_driver]

                    # Parse each of the two tables of the driver
                    for l_table, r_table in zip(table_separators[0:-1], table_separators[1:]):
                        # Refine table bottom
                        """
                        `b_tables` above is the shared table bottom of all tables in this row. Here
                        we want to get a more precised table bottom for this very table, so the
                        table rows can be more accurately located. Note that we use `b_tables + 10`
                        below when searching for table bottom. This buffer is added to make sure
                        the white strip is included.
                        """
                        if white_strip := page.search_for_white_strips(
                            clip=(l_table, b_table_header + 1, r_table, b_tables + 10),
                            height=lap_times_height / 3
                        ):
                            b_table = white_strip[0] + 1
                        else:
                            doc.close()
                            raise ParsingError(
                                f'Could not find any white strip below the table in ('
                                f'{l_table:.2f}, {b_table_header:.2f}, {r_table:.2f}, '
                                f'{b_tables:.2f}) on {page_no_str}'
                            )

                        # Get table header
                        cols = self._detect_cols(
                            page,
                            clip=(l_table, t_table_header, r_table, b_table_header),
                            col_min_gap=3,
                            min_black_line_length=0.5
                        )
                        if ((len(cols) != 2)  # noqa: PLR2004
                                or (cols[0].text != 'NO')
                                or (cols[1].text != 'TIME')):
                            raise ParsingError(
                                f'Expected "NO" and "TIME" cols. in ({l_table:.2f}, '
                                f'{t_table_header:.2f}, {r_table:.2f}, {b_table_header:.2f}) on '
                                f'{page_no_str}. Got: {cols}'
                            )
                        l_table = cols[0].l - 1  # More accurate left boundary  # noqa: PLW2901

                        # Vertical lines separating the two cols.
                        vlines = [l_table, cols[0].r, (cols[0].r + cols[1].l) / 2, r_table]

                        # Horizontal lines are located by the grey and white rows
                        hlines = page.search_for_grey_white_rows(
                            clip=(l_table, b_table_header + 1, r_table, b_table),
                            min_height=np.mean([i.b - i.t for i in cols]) / 2,
                            min_width=0.5
                        )

                        # Parse the table
                        df = page.parse_table_by_grid(vlines=vlines,
                                                      hlines=hlines,
                                                      header_included=False)
                        df.columns = ['lap_no', 'pit', 'lap_time']
                        df['lap_time_deleted'] = df.lap_time.apply(lambda x: x.strikeout is True)
                        df = df.map(self._normalise_textblock)
                        df['pit'] = (df.pit == 'P')
                        df['car_no'] = car_no
                        dfs.append(df)

        # Clean up
        df = pd.concat(dfs, ignore_index=True)
        df = df[df.lap_no != 1]  # First lap's lap time is calendar time of the lap, not lap time
        df.car_no = df.car_no.astype(int)
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
                    # TODO: why no pit field??
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

        The output dataframe has cols. [driver No., laps completed, total time,
        finishing position, finishing status, fastest lap time, fastest lap speed, fastest lap No.]
        """
        # Find the page with "Final Classification", on which the table is located
        doc = pymupdf.open(self.classification_file)
        page: Page
        classification: Optional[list[TextBlock]] = None
        for page in doc:
            page = Page(page, file=self.classification_file)  # noqa: PLW2901
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
        page_no_str = f'p.{page.number} in {page.file}'

        # Bottom position of "Final Classification", below which is the table header/col. names
        b_classification = classification[0].y1

        # Locate the first long black horizontal line below "Final Classification". This is the
        # line separating the table header and table body
        if black_lines := page.search_for_black_lines(clip=(0, b_classification, page.w, page.h)):
            t_table_body = black_lines[0]
        else:
            doc.close()
            raise ParsingError(f'Cannot find the black line separating table header and table '
                               f'body below "Final Classification" on {page_no_str}')

        # Get cols.
        cols = self._detect_cols(page,
                                 clip=(0, b_classification + 1, page.w, t_table_body - 1),
                                 col_min_gap=1)
        if not cols:
            doc.close()
            raise ParsingError(f'Could not locate cols. in the table header on {page_no_str}')
        if [i.text.upper() for i in cols] != ['NO', 'DRIVER', 'NAT', 'ENTRANT', 'LAPS', 'TIME',
                                              'GAP', 'INT', 'KM/H', 'FASTEST', 'ON', 'PTS']:
            doc.close()
            raise ParsingError(
                f'Got unexpected or miss some cols. on {page_no_str}. Expected "NO", "DRIVER", '
                f'"NAT", "ENTRANT", "LAPS", "TIME", "GAP", "INT", "KM/H", "FASTEST", "ON", and '
                f'"PTS". Got: {[i.text for i in cols]}'
            )
        vlines = [0,
                  cols[0].bbox[0] - 1,
                  (cols[0].bbox[2] + cols[1].bbox[0]) / 2,
                  cols[2].bbox[0] - 1,
                  (cols[2].bbox[2] + cols[3].bbox[0]) / 2,
                  cols[4].bbox[0],
                  cols[4].bbox[2],
                  (cols[5].bbox[2] + cols[6].bbox[0]) / 2,
                  (cols[6].bbox[2] + cols[7].bbox[0]) / 2,
                  (cols[7].bbox[2] + cols[8].bbox[0]) / 2,
                  (cols[8].bbox[2] + cols[9].bbox[0]) / 2,
                  (cols[9].bbox[2] + cols[10].bbox[0]) / 2,
                  (cols[10].bbox[2] + cols[11].bbox[0]) / 2,
                  cols[11].bbox[2] + 1]
        col_row_height = np.mean([i.bbox[3] - i.bbox[1] for i in cols])

        # Find the first white strip below the table header. This is the bottom of the table
        if white_strips := page.search_for_white_strips(clip=(0, t_table_body, page.w, page.h),
                                                       height=col_row_height / 2):
            b_table = white_strips[0] + 1
        else:
            doc.close()
            raise ParsingError(f'Could not find table bottom by white strip on {page_no_str}')

        # Horizontal lines separating the rows
        hlines = page.search_for_grey_white_rows(clip=(0, t_table_body + 1, page.w, b_table + 1),
                                                 min_height=col_row_height / 2)

        # Parse the table using the grid above
        df = page.parse_table_by_grid(vlines=vlines, hlines=hlines, header_included=False,
                                      allow_multiple_texts_per_cell=[2, 4])
        df.columns = ['finishing_position', 'car_no', 'driver', 'nat', 'team', 'laps_completed',
                      'time', 'gap', 'int', 'avg_speed', 'fastest_lap_time', 'fastest_lap_no',
                      'points']
        df = df.map(self._normalise_textblock, merge_multi_tbs=True)  # TODO: `merge_multi_tbs`
        """
        `allow_multiple_texts_per_cell` is turned on for string cols.: driver name and team name.
        Because if the input PDF is an image, e.g. 2025 Austrian race, then OCR may break a phrase
        into two or more text blocks, e.g. "George RUSSELL" to "George" and "RUSSELL". We don't
        need these cols. anyways, so just allow multiple text blocks in those cols., and also allow
        `merge_multi_tbs` in `self._normalise_textblock` to merge them into one string.
        """

        # Check if there is a "NOT CLASSIFIED" table below the main table
        not_classified = page.search_for('NOT CLASSIFIED', clip=(0, b_table + 1, page.w, page.h))

        # If yes, repeat the above for the "NOT CLASSIFIED" table
        if not_classified:
            if black_lines := page.search_for_black_lines(
                    clip=(0, not_classified[0].y1, page.w, page.h)
            ):
                t_table_body = black_lines[0] + 1
            else:
                doc.close()
                raise ParsingError(f'Cannot find the black line separating table header and table '
                                   f'body for "NOT CLASSIFIED" on {page_no_str}')
            if white_strips := page.search_for_white_strips(
                    clip=(0, t_table_body, page.w, page.h),
                    height=col_row_height / 2
            ):
                b_table = white_strips[0] + 1
            else:
                doc.close()
                raise ParsingError(
                    f'Could not find the bottom of "NOT CLASSIFIED" table by white strip on '
                    f'{page_no_str}'
                )
            hlines = page.search_for_grey_white_rows(clip=(0, t_table_body, page.w, b_table),
                                                     min_height=col_row_height / 3)
            not_classified = page.parse_table_by_grid(vlines=vlines,
                                                      hlines=hlines,
                                                      header_included=False,
                                                      allow_multiple_texts_per_cell=[2, 4])
            not_classified.columns = df.columns
            not_classified.position = None  # No finishing position for unclassified drivers
            not_classified = not_classified.map(self._normalise_textblock, merge_multi_tbs=True)
        else:
            # No unclassified drivers
            not_classified = pd.DataFrame(columns=df.columns)

        df['is_classified'] = True # Set all drivers from the main table as classified
        not_classified['finishing_status'] = 11  # TODO: should clean up the code later
        not_classified['is_classified'] = False
        df = pd.concat([df, not_classified], ignore_index=True)

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
        page: Page
        for page in doc:
            # Each page can have multiple (usually five) tables, all of which begins from the same
            # top y-position. The table headers are vertically bounded between "History Chart" and
            # a black line below the table header
            page = Page(page, file=self.history_chart_file)  # noqa: PLW2901
            page_no_str = f'p.{page.number} in {page.file}'
            history_chart = page.search_for('History Chart', clip=(0, 0, page.w, page.h / 3))
            if len(history_chart) != 1:
                doc.close()
                raise ParsingError(f'Find none or multiple "History Chart" on {page_no_str}')
            t_table_header = history_chart[0].y1 + 1
            if black_lines := page.search_for_black_lines(
                    clip=(0, t_table_header, page.w, page.h),
                    min_length=0.15  # Five tables in total, so each table occupies 20% of the page
            ):                       # width. We give 5% buffer, so `min_length` is set to 15%
                b_table_header = black_lines[0] + 1
            else:
                doc.close()
                raise ParsingError(f'Cannot find any black line below "History Chart" on '
                                   f'{page_no_str}')

            # All tables end above a sizable white strip
            if white_strips := page.search_for_white_strips(
                    clip=(0, b_table_header + 1, page.w, page.h)
            ):
                b_tables = white_strips[0] + 1
            else:
                doc.close()
                raise ParsingError(f'Cannot find any white strip below tables on {page_no_str}')

            # All side-by-side tables are separated by vertical white strips. Locate those strips
            page.set_rotation(90)
            if white_strips := page.search_for_white_strips(
                    clip=(page.h - b_tables, 0, page.h - b_table_header, page.w)
            ):
                table_separators = white_strips
            else:
                doc.close()
                raise ParsingError(f'Cannot find any vertical white strips between tables on '
                                   f'{page_no_str}')
            page.set_rotation(0)

            # Iterate over each table
            for l_table, r_table in zip(table_separators[:-1], table_separators[1:]):
                l_table += 1  # Some buffer so have a bit margins between tables  # noqa: PLW2901
                r_table += 1  # noqa: PLW2901
                table_clip_str = f'({l_table:.1f}, {b_table_header:.1f}, {r_table:.1f}, ' \
                                 f'{b_tables:.1f})'

                # Find table bottom
                if white_strips := page.search_for_white_strips(
                        clip=(l_table, b_table_header, r_table, b_tables + 10)
                ):
                    b_table = white_strips[0] + 1
                else:
                    doc.close()
                    raise ParsingError(f'Cannot not find any white strip below the table '
                                       f'{table_clip_str} on {page_no_str}')

                # Get cols.
                cols = self._detect_cols(
                    page,
                    clip=(l_table, t_table_header, r_table, b_table_header),
                    col_min_gap=2,  # Col. names are quite far from each other, so use a larger gap
                    min_black_line_length=0.6
                )
                if len(cols) != 3:  # noqa: PLR2004
                    raise ParsingError(f'Expected three cols. in the table {table_clip_str} on '
                                       f'{page_no_str}. Found: {cols}')
                if match := re.search(r'LAP (\d+)', cols[0].text):
                    lap_no = int(match.group(1))
                else:
                    doc.close()
                    raise ParsingError(f'Expected "LAP x" to be the zero-th col. of table '
                                       f'{table_clip_str} on {page_no_str}. Found: {cols[0]}')
                if (cols[1].text != 'GAP') or (cols[2].text != 'TIME'):
                    doc.close()
                    raise ParsingError(f'Expected "GAP" and "TIME" to be the first and second '
                                       f'col. of table {table_clip_str} on {page_no_str}. Found: '
                                       f'{cols[1:]}')
                line_height = np.mean([i.b - i.t for i in cols])
                vlines = [cols[0].l,
                          (cols[0].r + cols[1].l) / 2,
                          cols[1].l / 4 + cols[1].r / 4 + cols[2].l / 2,  # Midpoint between "GAP"
                          r_table]                                        # and "TIME", minus 1/4
                                                                          # width of "GAP"
                # Get row positions by white/grey background colours
                hlines = page.search_for_grey_white_rows(
                    clip=(l_table, b_table_header, r_table, b_table),
                    min_height=line_height / 3
                )

                # Parse the table using the grid above
                df = page.parse_table_by_grid(vlines=vlines, hlines=hlines, header_included=False)
                df.columns = [i.text.split()[0].lower() for i in cols]  # "LAP 5" --> "lap"
                df = df.rename(columns={'lap': 'car_no'})
                df = df.map(self._normalise_textblock)
                df['lap'] = lap_no

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
            lambda x: int(re.findall(r'\d+', x)[0]) if pd.notna(x) and ('LAP' in x) else 0
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
        page: Page
        for page in doc:
            page = Page(page, file=self.lap_chart_file)  # noqa: PLW2901
            page_no_str = f'p.{page.number} in {page.file}'

            # Table header/col. names are below "Race Lap Chart"
            if lap_chart := page.search_for('Lap Chart'):
                t_table_header = lap_chart[0].y1 + 1
            else:
                doc.close()
                raise ParsingError(f'Cannot find "Lap Chart" on p.{page_no_str}')

            # Table header is above a black line
            if black_lines := page.search_for_black_lines(
                    clip=(0, t_table_header, page.w, page.h)
            ):
                t_table_body = black_lines[0]
            else:
                raise ParsingError(f'Cannot find any black line below the table header '
                                   f'on {page_no_str}')

            # Table bottom is a white strip
            """
            A white strip of any height is fine. Because the table always has a black vertical line
            between col. 0 and col. 1, so there is no white strip at all in the table. Any white
            strip must indicate the end of the table.
            """
            if white_strips := page.search_for_white_strips(clip=(0, t_table_body, page.w, page.h),
                                                            height=1):
                b_table = white_strips[0] + 1
            else:
                doc.close()
                raise ParsingError(f'Cannot find any white strip below the table on {page_no_str}')

            # Get cols.
            cols = self._detect_cols(page,
                                     clip=(0, t_table_header, page.w, t_table_body - 1),
                                     col_min_gap=3,  # Col. names are quite far from each other
                                     min_black_line_length=0.5)
            if len(cols) <= 1:
                doc.close()
                raise ParsingError(f'Expected at least two cols. on {page_no_str}. Found: {cols}')
            if cols[0].text != 'POS':
                doc.close()
                raise ParsingError(f'Expected "POS" to be the zero-th col. on {page_no_str}. '
                                   f'Found: {cols[0]}')
            for i in range(1, len(cols)):
                if not re.match(r'^\d+$', cols[i].text):
                    doc.close()
                    raise ParsingError(f'Expected the {i}-th col. to be a number on '
                                       f'{page_no_str}. Found: {cols[i]}')

            # Find a black vertical line below the black horizontal line above. This separates the
            # zero-th col. and the first col.
            """
            `Page.search_for_black_lines` is for horizontal lines only. So to search for vertical
            lines, we rotate the page, i.e. applying [[0, -1], [1, 0]].
            """
            page.set_rotation(270)
            black_lines = page.search_for_black_lines(clip=(t_table_body, 0, b_table, page.w),
                                                      min_length=0.7)
            black_lines = [page.w - i for i in black_lines]  # Transpose back
            page.set_rotation(0)
            if len(black_lines) != 1:
                raise ParsingError(f'Cannot find or find multiple vertical black lines below the '
                                   f'table header on {page_no_str}: {black_lines}')
            l_first_col = black_lines[0]
            vlines = [0,
                      l_first_col,
                      *[(cols[i].r + cols[i + 1].l) / 2 for i in range(1, len(cols) - 1)],
                      cols[-1].r + 1]

            # Locate rows by POS col.
            """
            There is no background colour to indicate the rows, so we brute force the row positions
            by looking at "POS" col., i.e. the positions of "GRID", "LAP 1", "LAP 2", etc. We can
            transpose the page and then use `._detect_cols` to get the positions. We don't care
            about the text, because the page is transposed, so of course the texts will be wrong,
            but the positions of the text are correct. We use a small `col_min_gap`, because after
            transpose, the char. width will become char. height, so the gap needs to be adjusted
            accordingly.
            """
            page.set_rotation(270)
            rows = self._detect_cols(
                page,
                clip=(t_table_body + 1, page.w - l_first_col - 1, b_table, page.w),
                col_min_gap=0.3
            )
            page.set_rotation(0)
            if not rows:
                doc.close()
                raise ParsingError(f'Cannot detect any rows in the zero-th col. on {page_no_str}')
            hlines = [t_table_body,
                      *[(rows[i].r + rows[i + 1].l) / 2 for i in range(len(rows) - 1)],
                      b_table]

            # Parse the table
            df = page.parse_table_by_grid(vlines=vlines,
                                          hlines=hlines,
                                          header_included=False,
                                          tol=3)
            df = df.map(self._normalise_textblock)

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
        # TODO: this is identical to FP lap analysis and quali. lap times parser. Consider
        #       refactor/unify them
        doc = pymupdf.open(self.lap_analysis_file)
        dfs = []
        page: Page
        for page in doc:
            # Find "Lap Times"
            page = Page(page, file=self.lap_analysis_file)  # noqa: PLW2901
            page_no_str = f'p.{page.number} in {page.file}'
            lap_analysis = page.search_for('Lap Analysis')
            if len(lap_analysis) != 1:
                doc.close()
                raise ParsingError(f'Find none or multiple "Lap Analysis" on {page_no_str}')
            b_lap_times = lap_analysis[0].y1
            # Will use height of "Lap Analysis" as a reference for the white strips between drivers
            lap_analysis_height = lap_analysis[0].bbox[3] - lap_analysis[0].bbox[1]

            # Find the white strip immediately below "Lap Times", below which are the tables
            if white_strips := page.search_for_white_strips(clip=(0, b_lap_times, page.w, page.h)):
                t_all_drivers = white_strips[0]
            else:
                doc.close()
                raise ParsingError(f'Expect at least a white strip below "Lap Analysis" on '
                                   f'{page_no_str}. Found: {white_strips}')

            # Find all black horizontal lines
            black_lines = page.search_for_black_lines(clip=(0, t_all_drivers, page.w, page.h),
                                                      min_length=0.125)
            if not black_lines:
                doc.close()
                raise ParsingError(f'Cannot find any black line below "Lap Analysis" on '
                                   f'{page_no_str}')

            # Exclude the bottommost black line, which is the footnote separator, not a table
            bottom_black_line: Optional[float] = None
            if long_black_lines := page.search_for_black_lines(
                    clip=(0, black_lines[0], page.w, page.h),
                    min_length=0.7
            ):
                black_lines = [l for l in black_lines
                               if not any(np.isclose(l, long_black_line, atol=5)
                                          for long_black_line in long_black_lines)]
                bottom_black_line = long_black_lines[0]
            if not black_lines:
                doc.close()
                raise ParsingError(f'Cannot find any black line below "Lap Analysis" on '
                                   f'{page_no_str}')

            # Find vertical white spaces separating the three side-by-side drivers. These white
            # strips should be relatively tall. We use half of the "Lap Analysis" height as the
            # threshold here
            if len(black_lines) == 1:    # If only one row of drivers on the page, then the bottom
                b_page_content = page.h  # of the search area is page bottom, excl. footnote
                if bottom_black_line:
                    b_page_content = bottom_black_line - 1
                    if page_no_text := page.search_for(
                            f'Page {page.number + 1} of',
                            clip=(0, black_lines[-1], page.w, b_page_content)
                    ):
                        b_page_content = min(b_page_content, page_no_text[0].y0 - 1)
                clip = (page.h - b_page_content, 0, page.h - black_lines[0] - 1, page.w)
            else:
                clip = (page.h - black_lines[-1] + 1, 0, page.h - black_lines[0] - 1, page.w)
            page.set_rotation(90)
            driver_separators = page.search_for_white_strips(clip=clip,
                                                             height=lap_analysis_height * 0.5)
            page.set_rotation(0)
            # A driver has at least one table, so at least two white strips: one to the left of the
            # table and the other to the right of it
            if len(driver_separators) < 2:  # noqa: PLR2004
                doc.close()
                raise ParsingError(f'Expect at least two vertical white strips separating drivers '
                                   f'on {page_no_str}. Found: {driver_separators}')
            elif len(driver_separators) > 4:  # noqa: PLR2004
                doc.close()
                raise ParsingError(f'Expect at most four white strips separating drivers on '
                                   f'{page_no_str}. Found: {driver_separators}')

            # Each line should be the separator between a table header and its body, so use these
            # black lines to locate the tables
            for black_line in black_lines:
                # Table header is vertically between the first and second white strips above the
                # black line
                white_strips = page.search_for_white_strips(clip=(0, 0, page.w, black_line),
                                                            height=lap_analysis_height / 3)
                if len(white_strips) < 2:  # noqa: PLR2004
                    doc.close()
                    raise ParsingError(f'Found one or no white strip above the black line at '
                                       f'{black_line} on {page_no_str}: {white_strips}. Expected '
                                       f'at least two')
                t_driver = white_strips[-2] + 1
                t_table_header = white_strips[-1] + 1

                # Bottom of the table header is the very thin white strip above the black line
                if white_strip := page.search_for_white_strips(
                        clip=(0, t_table_header, page.w, black_line),
                        height=1
                ):
                    b_table_header = white_strip[-1] + 1
                else:
                    doc.close()
                    raise ParsingError(f'Cannot find any white strip separating table header '
                                       f'and body around {black_line} on {page_no_str}')

                # The shared bottom of all tables in the row is the white strip below black line
                if white_strips := page.search_for_white_strips(
                        clip=(0, black_line, page.w, page.h)
                ):
                    b_tables = white_strips[0] + 1
                else:
                    doc.close()
                    raise ParsingError(f'Cannot find any white strip below the black line at '
                                       f'{black_line} on {page_no_str}')

                # Parse each of the three side by side drivers' tables
                for l_driver, r_driver in zip(driver_separators[:-1], driver_separators[1:]):
                    l_driver += 1  # noqa: PLW2901
                    r_driver += 1  # noqa: PLW2901

                    # Get the driver name and car No.
                    driver = page.get_text(clip=(l_driver, t_driver, r_driver, t_table_header))
                    # Skip if no driver. E.g., four tables on a page. The second row only has one
                    # table, so should skip two in the second row
                    if (not driver) or (not ''.join(i.text for i in driver)):
                        continue
                    if match := re.match(r'^(\d+)\s+[A-Za-z ]+$', driver[0].text):
                        car_no = int(match.group(1))
                    else:
                        doc.close()
                        raise ParsingError(f'Could not parse car No. in '
                                           f'({l_driver:.1f}, {t_driver:.1f}, {t_driver:.1f}, '
                                           f'{t_table_header:.1f}) on {page_no_str}: {driver}')

                    # Find the thin vertical white strip separating two tables of the driver
                    page.set_rotation(90)
                    table_separators = page.search_for_white_strips(
                        clip=(page.h - b_tables, l_driver, page.h - t_table_header, r_driver),
                        height=1
                    )
                    page.set_rotation(0)
                    if np.isclose(table_separators[0], l_driver, atol=5):   # Don't need driver
                        table_separators.pop(0)                             # separators here. Will
                    if np.isclose(table_separators[-1], r_driver, atol=5):  # add back manually
                        table_separators.pop(-1)                            # later
                    if table_separators:
                        table_separators.insert(0, l_driver)
                        table_separators.append(r_driver)
                    else:
                        table_separators = [l_driver, r_driver]

                    # Parse each of the two tables of the driver
                    for l_table, r_table in zip(table_separators[0:-1], table_separators[1:]):
                        # Refine table bottom
                        if white_strip := page.search_for_white_strips(
                                clip=(l_table, b_table_header + 1, r_table, b_tables + 10),
                                height=lap_analysis_height / 3
                        ):
                            b_table = white_strip[0] + 1
                        else:
                            doc.close()
                            raise ParsingError(
                                f'Cannot find any white strip below the table in ('
                                f'{l_table:.1f}, {b_table_header:.1f}, {r_table:.1f}, '
                                f'{b_tables:.1f}) on {page_no_str}'
                            )

                        # Get table header
                        cols = self._detect_cols(
                            page,
                            clip=(l_table, t_table_header, r_table, b_table_header),
                            col_min_gap=3,
                            min_black_line_length=0.5
                        )
                        if ((len(cols) != 2)  # noqa: PLR2004
                                or (cols[0].text != 'LAP')
                                or (cols[1].text != 'TIME')):
                            raise ParsingError(
                                f'Expected "LAP" and "TIME" cols. in ({l_table:.1f}, '
                                f'{t_table_header:.1f}, {r_table:.1f}, {b_table_header:.1f}) on '
                                f'{page_no_str}. Got: {cols}'
                            )
                        l_table = cols[0].l - 1  # More accurate left boundary  # noqa: PLW2901

                        # Vertical lines separating the two cols.
                        vlines = [l_table, cols[0].r, (cols[0].r + cols[1].l) / 2, r_table]

                        # Horizontal lines are located by the grey and white rows
                        hlines = page.search_for_grey_white_rows(
                            clip=(l_table, b_table_header + 1, r_table, b_table),
                            min_height=np.mean([i.b - i.t for i in cols]) / 2,
                            min_width=0.5
                        )

                        # Skip if no lap, e.g. Gasly DNS in 2024 Silverstone, so no lap for him
                        if not hlines:
                            warnings.warn(f'No lap found for {driver[0].text.replace("\n", ' ')}. '
                                          f'Please check if this is correct, e.g. DNS')
                            continue

                        # Parse the table
                        df = page.parse_table_by_grid(vlines=vlines,
                                                      hlines=hlines,
                                                      header_included=False)
                        df.columns = ['lap', 'pit', 'lap_time']
                        df['lap_time_deleted'] = df.lap_time.apply(lambda x: x.strikeout is True)
                        df = df.map(self._normalise_textblock)
                        df = df.dropna(subset='lap')  # E.g. Gasly 2024 Silverstone
                        if df.empty:
                            warnings.warn(f'No lap found for {driver[0].text.replace("\n", ' ')}. '
                                          f'Please check if this is correct, e.g. DNS')
                            continue
                        df['pit'] = (df.pit == 'P')
                        df['car_no'] = car_no
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
            df.to_json = partial(self._quali_lap_times_to_json, df=df,
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
        page: Page
        for i in range(len(doc)):
            page = Page(doc[i], file=self.classification_file)  # noqa: PLW2901
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
            raise ParsingError(f'"Final Classification" or "Provisional Classification" not found '
                               f'on any page in {self.classification_file}')

        # Bottom of "Final Classification"
        page_no_str = f'p.{page.number} in {self.classification_file}'
        b_classification = classification[0].y1 + 1

        # First black horizontal line below "Classification" separates table header and table body
        if black_lines := page.search_for_black_lines(clip=(0, b_classification, page.w, page.h)):
            t_table_body = black_lines[0]
        else:
            doc.close()
            raise ParsingError(f'Cannot find the black line separating table header and table '
                               f'body below "Final Classification" on {page_no_str}')

        # Get cols.
        cols = self._detect_cols(page,
                                 clip=(0, b_classification + 1, page.w, t_table_body - 1),
                                 col_min_gap=1)
        if not cols:
            doc.close()
            raise ParsingError(f'Cannot locate col. names in the table header on {page_no_str}')
        if self.session == 'sprint_quali':  # Always use "Q1", "Q2", and "Q3" for both quali. and
            for col in cols:           # sprint quali.
                if col.text == 'SQ1':
                    col.text = 'Q1'
                elif col.text == 'SQ2':
                    col.text = 'Q2'
                elif col.text == 'SQ3':
                    col.text = 'Q3'
        for col_name in ['LAPS', 'TIME']:  # Add prefix to "LAPS" and "TIME" for each session
            session = 1
            for col in cols:
                if col.text == col_name:
                    col.text = f'Q{session}_{col.text}'
                    session += 1
        col_name_to_tb = {i.text: i for i in cols}  # Col. name --> its TextBlock
        col_row_height = np.mean([i.bbox[3] - i.bbox[1] for i in cols])

        # The table ends with a white strip below the table header
        if white_strips:= page.search_for_white_strips(clip=(0, t_table_body, page.w, page.h),
                                                       height=col_row_height / 3):
            b_table = white_strips[0]
        else:
            doc.close()
            raise ParsingError(f'Cannot find table bottom by white strip on {page_no_str}')

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
        cols = self._detect_cols(page,
                                 clip=(col_name_to_tb['ENTRANT'].l - 1,
                                       t_table_body + 1,
                                       col_name_to_tb['Q1_LAPS'].l,
                                       b_table + 1),
                                 col_min_gap=1)
        if not (len(cols) == 2 and re.match(r'[\d:\n.]+', cols[1].text)):  # noqa: PLR2004
            doc.close()
            raise ParsingError(f'Cannot locate the boundary between ENTRANT and Q1 on '
                               f'{page_no_str}')
        sep_entrant_q1 = (cols[0].r + cols[1].l) / 2
        # Boundary between Q1 % and Q1_TIME
        if '%' in col_name_to_tb:
            cols = self._detect_cols(page,
                                     clip=(col_name_to_tb['%'].l,
                                           t_table_body + 1,
                                           col_name_to_tb['Q1_TIME'].l,
                                           b_table + 1),
                                     col_min_gap=1.1)
            if not (len(cols) == 2  # noqa: PLR2004
                    and re.match(r'[\d.]+', cols[0].text)
                    and re.match(r'[\d.]+', cols[1].text)):
                doc.close()
                raise ParsingError(f'Could not locate the boundary between Q1 % and Q1 TIME on '
                                   f'{page_no_str}: {cols}')
            sep_q1_pct_q1time = (cols[0].r + cols[1].l) / 2
        # Boundary between Q1_TIME and Q2
        page.set_rotation(90)
        black_lines = page.search_for_black_lines(clip=(page.h - b_table,
                                                  col_name_to_tb['Q1_TIME'].r,
                                                  page.h - t_table_body,
                                                  col_name_to_tb['Q2'].l),
                                                  scaling_factor=16)
        if len(black_lines) != 1:
            doc.close()
            raise ParsingError(f'Cannot locate the boundary between Q1 TIME and Q2 on '
                               f'{page_no_str}: {black_lines}')
        sep_q1time_q2 = black_lines[0]
        # Boundary between Q2_TIME and Q3
        black_lines = page.search_for_black_lines(clip=(page.h - b_table,
                                                        col_name_to_tb['Q2_TIME'].r,
                                                        page.h - t_table_body,
                                                        col_name_to_tb['Q3'].l),
                                                  scaling_factor=16)
        if len(black_lines) != 1:
            raise ParsingError(f'Cannot locate the boundary between Q2 TIME and Q3 on '
                               f'{page_no_str}: {black_lines}')
        sep_q2time_q3 = black_lines[0]
        page.set_rotation(0)
        # All col. positions
        vlines = [0,
                  col_name_to_tb['NO'].l - 1,
                  (col_name_to_tb['NO'].r + col_name_to_tb['DRIVER'].l) / 2,
                  col_name_to_tb['NAT'].l - 1,
                  (col_name_to_tb['NAT'].r + col_name_to_tb['ENTRANT'].l) / 2,
                  sep_entrant_q1,
                  col_name_to_tb['Q1_LAPS'].l,
                  col_name_to_tb['Q1_LAPS'].r,
                  sep_q1time_q2,
                  col_name_to_tb['Q2_LAPS'].l,
                  col_name_to_tb['Q2_LAPS'].r,
                  sep_q2time_q3,
                  col_name_to_tb['Q3_LAPS'].l,
                  col_name_to_tb['Q3_LAPS'].r,
                  page.w]
        if '%' in col_name_to_tb:
            # TODO: this assumes dict is ordered. Not sure which Python version started this
            vlines.insert(list(col_name_to_tb.keys()).index('%') + 2, sep_q1_pct_q1time)

        # Row positions
        hlines = page.search_for_grey_white_rows(clip=(0, t_table_body, page.w, b_table),
                                                 min_height=col_row_height / 3)

        # Get the table
        df = page.parse_table_by_grid(vlines=vlines, hlines=hlines, header_included=False)
        df.columns = ['position'] + [i for i in col_name_to_tb.keys()]
        df['finishing_status'] = 0
        df['original_order'] = range(1, len(df) + 1)  # Driver's original order in the PDF
        df['is_classified'] = True

        # Parse "NOT CLASSIFIED" table, if any
        if not_classified := page.search_for('NOT CLASSIFIED', clip=(0, b_table, page.w, page.h)):
            t_table_body = not_classified[0].y1 + 1
            if white_strips:= page.search_for_white_strips(clip=(0, t_table_body, page.w, page.h),
                                                           height=col_row_height / 3):
                b_table = white_strips[0]
            else:
                doc.close()
                raise ParsingError(f'Cannot find the bottom of "NOT CLASSIFIED" table by white '
                                   f'strip on {page_no_str}')
            hlines = page.search_for_grey_white_rows(clip=(0, t_table_body, page.w, b_table),
                                                     min_height=col_row_height / 3)
            not_classified = page.parse_table_by_grid(vlines=vlines,
                                                      hlines=hlines,
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
                if white_strips := page.search_for_white_strips(
                        clip=(0, t_table_body, page.w, page.h),
                        height=col_row_height / 3
                ):
                    b_table = white_strips[0]
                else:
                    doc.close()
                    raise ParsingError(f'Cannot find the bottom of "DISQUALIFIED" table by white '
                                       f'strip on {page_no_str}')
                hlines = page.search_for_grey_white_rows(clip=(0, t_table_body, page.w, b_table),
                                                         min_height=col_row_height / 3)
                disqualified = page.parse_table_by_grid(vlines=vlines,
                                                        hlines=hlines,
                                                        header_included=False)
                disqualified.columns = df.columns.drop(['finishing_status', 'original_order',
                                                        'is_classified'])
                disqualified.position = None  # No finishing position for DSQ drivers
                n = len(df)
                disqualified['original_order'] = range(n + 1, n + len(disqualified) + 1)
                disqualified['finishing_status'] = 20  # TODO: should clean up the code later
                disqualified['is_classified'] = False
            else:
                warnings.warn(f'Found "DISQUALIFIED" on {page_no_str}, but it is not horizontally '
                              f'centred. May be a penalty note instead of a DISQUALIFIED table. '
                              f'Ignored')
                disqualified = pd.DataFrame(columns=df.columns)
        else:
            disqualified = pd.DataFrame(columns=df.columns)
        df = pd.concat([df, disqualified], ignore_index=True)

        """
        `is_classified` here is simply a flag to indicate whether the driver belongs to the "NOT
        CLASSIFIED" table. It doesn't mean a driver is classified or not in F1 sense. Basically
        everyone in "NOT CLASSIFIED" table is not classified, and in addition, those in the main
        table who receive DSQ or DNQ are not classified either.
        """

        # Fill in the position for DNF and DSQ drivers
        # TODO: check this
        df = df.map(self._normalise_textblock)
        df.loc[df.position.isin(['DQ', 'DSQ']), 'finishing_status'] = 20
        df.loc[df.position != 'DQ', 'temp'] = df.position
        df.temp = df.temp.astype(float)
        df.loc[df.position != 'DQ', 'temp'] = df.loc[df.position != 'DQ', 'temp'].ffill() \
                                              + df.loc[df.position != 'DQ', 'temp'].isna().cumsum()
        df = df.sort_values(by='temp')
        df.temp = df.temp.ffill() + df.temp.isna().cumsum()
        df.position = df.temp.astype(int)
        del df['temp']
        df.NO = df.NO.astype(int)
        del df['NAT']

        # Overwrite `.to_json()` methods
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
            page_no_str = f'p.{page.number} in {self.lap_times_file}'
            quali_lap_times = page.search_for('Lap Times')
            if len(quali_lap_times) != 1:
                doc.close()
                raise ParsingError(f'Find none or multiple "Lap Times" on {page_no_str}')
            b_lap_times = quali_lap_times[0].y1
            # Will use height of "Lap Times" as a reference for the white strips between drivers
            lap_times_height = quali_lap_times[0].bbox[3] - quali_lap_times[0].bbox[1]

            # Find the white strip immediately below "Lap Times", below which are the tables
            if white_strips := page.search_for_white_strips(clip=(0, b_lap_times, page.w, page.h)):
                t_all_drivers = white_strips[0]
            else:
                doc.close()
                raise ParsingError(f'Expect at least a white strip below "Lap Times" on '
                                   f'{page_no_str}. Found none')

            # Find all black horizontal lines (see RaceParser._parse_lap_analysis for details)
            black_lines = page.search_for_black_lines(clip=(0, t_all_drivers, page.w, page.h),
                                                      min_length=0.125)
            if not black_lines:
                doc.close()
                raise ParsingError(f'Cannot find any black line below "Lap Times" on'
                                   f'{page_no_str}')

            # Exclude the bottommost black line, which is the footnote separator, not a table
            bottom_black_line: Optional[float] = None
            if long_black_lines := page.search_for_black_lines(
                    clip=(0, black_lines[0], page.w, page.h),
                    min_length=0.7
            ):
                black_lines = [l for l in black_lines
                               if not any(np.isclose(l, long_black_line, atol=5)
                                          for long_black_line in long_black_lines)]
                bottom_black_line = long_black_lines[0]
            if not black_lines:
                doc.close()
                raise ParsingError(f'Cannot find any black line below "Lap Analysis" on '
                                   f'{page_no_str}')

            # Find vertical white spaces separating the three side-by-side drivers. These white
            # strips should be relatively tall. We use half of the "Lap Analysis" height as the
            # threshold here
            if len(black_lines) == 1:    # If only one row of drivers on the page, then the bottom
                b_page_content = page.h  # of the search area is page bottom, excl. footnote
                if bottom_black_line:
                    b_page_content = bottom_black_line - 1
                    if page_no_text := page.search_for(
                            f'Page {page.number + 1} of',
                            clip=(0, black_lines[-1], page.w, b_page_content)
                    ):
                        b_page_content = min(b_page_content, page_no_text[0].y0 - 1)
                clip = (page.h - b_page_content, 0, page.h - black_lines[0] - 1, page.w)
            else:
                clip = (page.h - black_lines[-1] + 1, 0, page.h - black_lines[0] - 1, page.w)
            page.set_rotation(90)
            driver_separators = page.search_for_white_strips(clip=clip,
                                                             height=lap_times_height * 0.5)
            page.set_rotation(0)

            # A driver has at least one table, so at least two white strips: one to the left of the
            # table and the other to the right of it
            if len(driver_separators) < 2:  # noqa: PLR2004
                doc.close()
                raise ParsingError(f'Expect at least two vertical white strips separating drivers '
                                   f'on {page_no_str}. Found: {driver_separators}')
            elif len(driver_separators) > 4:  # noqa: PLR2004
                doc.close()
                raise ParsingError(f'Expect at most four white strips separating drivers on '
                                   f'{page_no_str}. Found: {driver_separators}')

            # Each line should be the separator between a table header and its body, so use these
            # black lines to locate the tables
            for black_line in black_lines:
                # Table header is vertically between the first and second white strips above the
                # black line
                white_strips = page.search_for_white_strips(clip=(0, 0, page.w, black_line),
                                                            height=lap_times_height / 3)
                if len(white_strips) < 2:  # noqa: PLR2004
                    doc.close()
                    raise ParsingError(f'Found one or no white strip above the black line at '
                                       f'{black_line} on {page_no_str}: {white_strips}. Expected '
                                       f'at least two')
                t_driver = white_strips[-2] + 1
                t_table_header = white_strips[-1] + 1

                # Bottom of the table header is the very thin white strip above the black line
                if white_strip := page.search_for_white_strips(
                        clip=(0, t_table_header, page.w, black_line),
                        height=1
                ):
                    b_table_header = white_strip[-1] + 1
                else:
                    doc.close()
                    raise ParsingError(f'Cannot find any white strip separating table header '
                                       f'and body around {black_line} on {page_no_str}')

                # The shared bottom of all tables in the row is the white strip below black line
                if white_strips := page.search_for_white_strips(
                        clip=(0, black_line, page.w, page.h)
                ):
                    b_tables = white_strips[0] + 1
                else:
                    doc.close()
                    raise ParsingError(f'Cannot find any white strip below the black line at '
                                       f'{black_line} on {page_no_str}')

                # Parse each of the three side by side drivers' tables
                for l_driver, r_driver in zip(driver_separators[:-1], driver_separators[1:]):
                    l_driver += 1  # noqa: PLW2901
                    r_driver += 1  # noqa: PLW2901

                    # Get the driver name and car No.
                    driver = page.get_text(clip=(l_driver, t_driver, r_driver, t_table_header))
                    # Skip if no driver. E.g., four tables on a page. The second row only has one
                    # table, so should skip two in the second row
                    if (not driver) or (not ''.join(i.text for i in driver)):
                        continue
                    if match := re.match(r'^(\d+)\s+[A-Za-z ]+$', driver[0].text):
                        car_no = int(match.group(1))
                    else:
                        doc.close()
                        raise ParsingError(f'Could not parse car No. in '
                                           f'({l_driver:.1f}, {t_driver:.1f}, {t_driver:.1f}, '
                                           f'{t_table_header:.1f}) on {page_no_str}: {driver}')

                    # Find the thin vertical white strip separating two tables of the driver
                    page.set_rotation(90)
                    table_separators = page.search_for_white_strips(
                        clip=(page.h - b_tables, l_driver, page.h - t_table_header, r_driver),
                        height=1
                    )
                    page.set_rotation(0)
                    if np.isclose(table_separators[0], l_driver, atol=5):  # Don't need driver
                        table_separators.pop(0)  # separators here. Will
                    if np.isclose(table_separators[-1], r_driver, atol=5):  # add back manually
                        table_separators.pop(-1)  # later
                    if table_separators:
                        table_separators.insert(0, l_driver)
                        table_separators.append(r_driver)
                    else:
                        table_separators = [l_driver, r_driver]

                    # Parse each of the two tables of the driver
                    for l_table, r_table in zip(table_separators[0:-1], table_separators[1:]):
                        # Refine table bottom
                        if white_strip := page.search_for_white_strips(
                                clip=(l_table, b_table_header + 1, r_table, b_tables + 10),
                                height=lap_times_height / 3
                        ):
                            b_table = white_strip[0] + 1
                        else:
                            doc.close()
                            raise ParsingError(
                                f'Cannot find any white strip below the table in ('
                                f'{l_table:.1f}, {b_table_header:.1f}, {r_table:.1f}, '
                                f'{b_tables:.1f}) on {page_no_str}'
                            )

                        # Get table header
                        cols = self._detect_cols(
                            page,
                            clip=(l_table, t_table_header, r_table, b_table_header),
                            col_min_gap=3,
                            min_black_line_length=0.5
                        )
                        if ((len(cols) != 2)  # noqa: PLR2004
                                or (cols[0].text != 'NO')
                                or (cols[1].text != 'TIME')):
                            raise ParsingError(
                                f'Expected "NO" and "TIME" cols. in ({l_table:.1f}, '
                                f'{t_table_header:.1f}, {r_table:.1f}, {b_table_header:.1f}) on '
                                f'{page_no_str}. Got: {cols}'
                            )
                        l_table = cols[0].l - 1  # More accurate left boundary  # noqa: PLW2901

                        # Vertical lines separating the two cols.
                        vlines = [l_table, cols[0].r, (cols[0].r + cols[1].l) / 2, r_table]

                        # Horizontal lines are located by the grey and white rows
                        hlines = page.search_for_grey_white_rows(
                            clip=(l_table, b_table_header + 1, r_table, b_table),
                            min_height=np.mean([i.b - i.t for i in cols]) / 2,
                            min_width=0.5
                        )

                        # Skip if no lap, e.g. Gasly DNS in 2024 Silverstone, so no lap for him
                        if not hlines:
                            warnings.warn(
                                f'No lap found for {driver[0].text.replace("\n", ' ')}. '
                                f'Please check if this is correct, e.g. DNS')
                            continue

                        # Parse the table
                        df = page.parse_table_by_grid(vlines=vlines,
                                                      hlines=hlines,
                                                      header_included=False)
                        df.columns = ['lap', 'pit', 'lap_time']
                        df['lap_time_deleted'] = df.lap_time.apply(
                            lambda x: x.strikeout is True)
                        df = df.map(self._normalise_textblock)
                        df = df.dropna(subset='lap')  # E.g. Tsunoda 2025 Imola
                        if df.empty:
                            warnings.warn(
                                f'No lap found for {driver[0].text.replace("\n", ' ')}. Please '
                                f'check if this is correct, e.g. DNS'
                            )
                            continue
                        df['pit'] = (df.pit == 'P')
                        df['car_no'] = car_no
                        dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        df = df.rename(columns={'lap': 'lap_no'})
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
        df.to_json = partial(self._quali_lap_times_to_json,
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

    @staticmethod
    def _quali_lap_times_to_json(df, year, round_no, session) -> list[dict]:
        # TODO: Very bad. Why did I create this method???
        lap_data = []
        # TODO: first lap's lap time is calendar time, not lap time, so drop it
        # Lap No. can be missing (e.g.#47)
        df = df[(df.lap_no >= 2) | df.lap_no.isna()].copy()  # noqa: PLR2004
        df.lap_time = df.lap_time.apply(duration_to_millisecond)
        for q in [1, 2, 3]:
            temp = df[df.Q == q].copy()
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
            temp['session_entry'] = temp['car_no'].map(
                lambda x: SessionEntryForeignKeys(
                    year=year,
                    round=round_no,
                    session=f'Q{q}' if session == 'quali' else f'SQ{q}',
                    car_number=x
                )
            )
            temp['lap_data'] = temp.apply(
                lambda x: LapImport(
                    object_type="Lap",
                    foreign_keys=x['session_entry'],
                    objects=x['lap']
                ).model_dump(exclude_unset=True),
                axis=1
            )
            lap_data.extend(temp['lap_data'].tolist())
        return lap_data


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
            page_no_str = f'p.{page.number} in {self.file}'

            # Locate "Pit Stop Summary" title
            pit_stop_summary = page.search_for('Pit Stop Summary')
            if len(pit_stop_summary) != 1:
                raise ParsingError(f'Find none or multiple "Pit Stop Summary" on {page_no_str}')
            b_title = pit_stop_summary[0].y1

            # Locate table header, vertically between the topmost black line and the white strip
            # immediately above the line
            black_lines = page.search_for_black_lines(clip=(0, b_title, page.w, page.h))
            if not black_lines:
                doc.close()
                raise ParsingError(f'Could not find a black horizontal line on {page_no_str}')
            b_table_header = black_lines[0] - 1
            t_table_body = black_lines[0] + 1
            if white_strips := page.search_for_white_strips(clip=(0, 0, page.w, b_table_header)):
                t_table_header = white_strips[-1] + 1
            else:
                doc.close()
                raise ParsingError(f'Could not find a white horizontal strip above the black line '
                                   f'on {page_no_str}')

            # Get col. names
            cols = self._detect_cols(page,
                                     clip=(0, t_table_header, page.w, b_table_header),
                                     col_min_gap=2)  # Very wide cols., so allow larger gaps
            if [i.text for i in cols] != ['NO', 'DRIVER', 'ENTRANT', 'LAP', 'TIME OF DAY', 'STOP',
                                          'DURATION', 'TOTAL TIME']:
                raise ParsingError(f'Found unexpected or less cols. on {page_no_str}: {cols}')

            # Table bottom is the first white strip below the table header
            if white_strips := page.search_for_white_strips(
                    clip=(0, t_table_body, page.w, page.h)
            ):
                b_table = white_strips[0] + 1
            else:
                doc.close()
                raise ParsingError(f'Could not find a white horizontal strip below the table '
                                   f'header on {page_no_str}')

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
            df = page.parse_table_by_grid(vlines=vlines, hlines=hlines, header_included=False)
            df.columns = [i.text for i in cols]
            dfs.append(df)

        # Clean up the table
        df = pd.concat(dfs, ignore_index=True)
        df = df.map(self._normalise_textblock)
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
