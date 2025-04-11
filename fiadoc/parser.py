# -*- coding: utf-8 -*-
import os
import pickle
import re
import warnings
from typing import Literal, get_args

import numpy as np
import pandas as pd
import pymupdf
from pydantic import ValidationError

from ._constants import QUALI_DRIVERS
from .models.classification import (
    SessionEntryImport,
    SessionEntryObject,
)
from .models.driver import RoundEntryImport, RoundEntryObject
from .models.foreign_key import PitStopForeignKeys, RoundEntry, SessionEntryForeignKeys
from .models.lap import LapImport, LapObject
from .models.pit_stop import PitStopData, PitStopObject
from .utils import Page, duration_to_millisecond, time_to_timedelta

pd.set_option('future.no_silent_downcasting', True)

RaceSessionT = Literal['race', 'sprint']
QualiSessionT = Literal['quali', 'sprint_quali']


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
            tol: float = 2
    ) -> pd.DataFrame:
        """Manually parse the table cell by cell, defined by lines separating the columns and rows

        The reason why we parse this table manually rather than using `page.find_tables` is that
        when we have the reserve driver table, car No. may have superscript, which can not be
        handled otherwise. The superscript indicates which reserve driver is driving whose car.
        E.g., Antonelli is driving Hamilton's car, then driver No. 44 and driver No. 12 will have
        the same superscript.

        :param vlines: x-coords. of vertical lines separating the cols.
        :param hlines: y-coords. of horizontal lines separating the rows
        :param tol: tolerance for bbox. of text. Default is 2 pixels. See #33
        """
        cells = []
        vgap = vlines[1] - vlines[0]  # Usual gap between two vertical lines
        for i in range(len(hlines) - 1):
            row = []
            superscripts = []
            for j in range(len(vlines) - 1):

                # Check if there is an unusual gap between two horizontal lines. If so, then we are
                # now at the gap between the main table and the reserve driver table
                if hlines[i + 1] - hlines[i] < vgap + 5:
                    t = hlines[i]
                    b = hlines[i + 1]
                else:
                    if i >= 1:  # The zero-th row is always fine
                        if hlines[i] - hlines[i - 1] < vgap + 5:  # The unusual big gap is below,
                            t = hlines[i]                         # so we are at the main table
                            b = hlines[i] + vgap + tol
                        else:  # The unusual big gap is above, so now at the reserve driver table
                            t = hlines[i + 1] - vgap - tol
                            b = hlines[i + 1]

                # Get text in the cell
                # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html
                """
                For each cell defined by the `vlines` and `hlines`, we get text inside it. However,
                texts that are partially inside the cell will also be captured by pymupdf. So we
                need to check whether the found text is indeed inside the cell's bbox, and do not
                false include other texts. However, we do want to allow a bit of tolerance, as
                `hlines` are not always perfect. The tolerance is set to 2 pixels in general. But
                for PDFs that have smaller page margin, i.e. texts and line height are bigger in
                these PDFs, we need to increase the tolerance.
                """
                cell = page.get_text(
                    'dict',
                    clip=(vlines[j], t, vlines[j + 1], b)
                )
                spans = []
                for block in cell['blocks']:
                    for line in block['lines']:
                        for span in line['spans']:
                            if span['text'].strip():
                                bbox = span['bbox']
                                # Need to check if the found text is indeed in the cell's bbox.
                                # PyMuPDF is notoriously bad for not respecting `clip` parameter.
                                # We give two pixels tolerance. See #33
                                if bbox[0] >= vlines[j] - tol \
                                        and bbox[2] <= vlines[j + 1] + tol \
                                        and bbox[1] >= t - tol \
                                        and bbox[3] <= b + tol:
                                    spans.append(span)

                # Check if any superscript
                # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-analyze-
                # font-characteristics for font flags
                superscript = None
                match len(spans):
                    case 1:  # Only one text so no superscript
                        row.append(spans[0]['text'].strip())
                    case 2:  # One text and one superscript
                        n_superscripts = 0
                        n_regular_text = 0
                        for span in spans:
                            match span['flags']:
                                case 0:
                                    regular_text = span['text'].strip()
                                    n_regular_text += 1
                                case 1:
                                    superscript = span['text'].strip()
                                    n_superscripts += 1
                                case _:
                                    raise ValueError(f'Unknown error when parsing row {i}, col '
                                                     f'{j} in {self.file}')
                        # If we found two regular text above, then have to decide which is the
                        # superscript using font size
                        if n_superscripts == 2 or n_regular_text == 2 or n_regular_text == 0:
                            temp = (spans[0]['size'] + spans[1]['size']) / 2
                            if (spans[0]['size'] - spans[1]['size']) / temp > 0.2:
                                superscript = spans[1]['text'].strip()
                                regular_text = spans[0]['text'].strip()
                            elif (spans[1]['size'] - spans[0]['size']) / temp > 0.2:
                                superscript = spans[0]['text'].strip()
                                regular_text = spans[1]['text'].strip()
                            # In principle superscript font size should be sufficiently smaller
                            # than the regular text. If the diff. is less than 20% raise an error
                            else:
                                raise ValueError(f'Cannot determine which text is superscript in '
                                                 f'row {i}, col {j} in {self.file}: {cell}')
                        row.append(regular_text)
                        superscripts.append(superscript)
                    case _:
                        raise ValueError(f'Unknown error when parsing row {i}, col {j} in '
                                         f'{self.file}: {cell}')

            # Only the first cell can have superscript, so after processing all cells in the
            # row, should only get one single superscript
            if len(superscripts) > 1:
                raise ValueError(f'Found multiple superscripts in row {i}, col {j} in '
                                 f'{self.file}: {cell}')
            row.append(superscripts[0] if superscripts else None)
            cells.append(row)

        # Convert to df.
        df = pd.DataFrame(cells)
        if df.shape[1] == 5:
            df.columns = ['car_no', 'driver', 'nat', 'team', 'constructor']
        elif df.shape[1] == 6:
            df.columns = ['car_no', 'driver', 'nat', 'team', 'constructor', 'reserve']
        else:
            raise ValueError(f'Expected 5 or 6 columns, got {df.shape[1]} in {self.file}')
        df.car_no = df.car_no.astype(int)
        assert df.car_no.is_unique, f'Car No. is not unique in {self.file}'

        # Clean up the reserve driver relationship
        if 'reserve' not in df.columns:
            df['reserve'] = False
            return df
        df['reserve_for'] = None
        for i in df.reserve.dropna().unique():
            temp = df[df.reserve == i]
            if len(temp) == 1:
                # handle the case where a driver is incorrectly indicated as
                # having a reserve driver, e.g. copy-paste error 2024, round 5
                df.loc[df.reserve == i, 'reserve'] = None
                warnings.warn(
                    f'Driver {temp.driver.iloc[0]} is indicated as being or having a reserve '
                    f'driver but no associated driver was found!'
                )
                continue

            assert len(temp) == 2, f'Expected 2 rows for superscript {i}, got {len(temp)}'
            assert temp.car_no.nunique() == 2, \
                f'Expected 2 different drivers for superscript {i}, got {temp.car_no.nunique()}'
            df.loc[df[df.reserve == i].index[1], 'reserve_for'] = temp.car_no.iloc[0]
        df.reserve = df['reserve_for'].notnull()
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
        delineating the bottom of the table. Below the table, we have stewards' name, so we can't
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

        # Lines separating the rows, which are the midpoints of the car No. texts
        aux_hlines = [(car_nos[i][3] + car_nos[i + 1][1]) / 2 for i in range(len(car_nos) - 1)]
        # Line vertically between "No." and the first car No.
        aux_hlines.insert(0, (headers['no'].y1 + car_nos[0][1]) / 2)
        # Line below the last car, which is last line + line gap (= last - 2nd last)
        aux_hlines.append(2 * aux_hlines[-1] - aux_hlines[-2])

        # Get the table
        tol = 2 if l > 40 else 3
        df = self._parse_table_by_grid(page, aux_vlines, aux_hlines, tol)

        def to_json() -> list[dict]:
            drivers = []
            for x in df.itertuples():
                try:
                    drivers.append(RoundEntryImport(
                            object_type="RoundEntry",
                            foreign_keys=RoundEntry(
                                year=self.year,
                                round=self.round_no,
                                team_reference=x.constructor,
                                driver_reference=x.driver
                            ),
                            objects=[
                                RoundEntryObject(
                                    car_number=x.car_no
                                )
                            ]
                        ).model_dump(exclude_unset=True))
                except ValidationError as e:
                    warnings.warn(f'Error when parsing driver {x.driver} in {self.file}: {e}')
            return drivers

        def to_pkl(filename: str | os.PathLike) -> None:
            with open(filename, 'wb') as f:
                pickle.dump(df.to_json(), f)

        df.to_json = to_json
        df.to_pkl = to_pkl
        return df


class RaceParser:
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
        self.classification_df = self._parse_classification()
        self.starting_grid = None  # By `_parse_lap_chart` in `self._parse_lap_times()`
        self.lap_times_df = self._parse_lap_times()
        # self._cross_validate()

    def _check_session(self) -> None:
        """Check that the input session is valid. Raise an error otherwise"""
        if self.session not in get_args(RaceSessionT):
            raise ValueError(f'Invalid session: {self.session}. '
                             f'Valid sessions are: {get_args(RaceSessionT)}')
        return

    # TODO: refactor. Have too many parse table by grid, in utils, in quali. parser, here, and in
    #       entry list
    def _parse_table_by_grid(
            self,
            page: Page,
            vlines: list[float],
            hlines: list[float],
            tol: float = 2
    ) -> pd.DataFrame:
        """Manually parse the table cell by cell, defined by lines separating the columns and rows

        See `EntryListParser._parse_table_by_grid()` for detailed explanation.

        :param vlines: x-coords. of vertical lines separating the cols. Table left and right
                       boundaries need to be included
        :param hlines: y-coords. of horizontal lines separating the rows. Table top and bottom
                       boundaries need to be included
        :param tol: tolerance for bbox. of text. Default is 2 pixels
        """
        cells = []
        for i in range(len(hlines) - 1):
            row = []
            for j in range(len(vlines) - 1):
                text = ''
                l, t, r, b = vlines[j], hlines[i], vlines[j + 1], hlines[i + 1]
                cell = page.get_text('blocks', clip=(l, t, r, b))
                if cell:
                    assert len(cell) == 1, f'Expected exactly one cell in row {i}, col {j} in ' \
                                           f'{self.classification_file}. Found {len(cell)}'
                    cell = cell[0]
                    if cell[4].strip():
                        bbox = cell[0:4]
                        if bbox[0] < vlines[j] - tol or bbox[2] > vlines[j + 1] + tol \
                                or bbox[1] < t - tol or bbox[3] > b + tol:
                            raise ValueError(f'Found text outside the cell in row {i}, col {j} in '
                                             f'{self.classification_file}')
                        text = cell[4].strip()
                row.append(text)
            cells.append(row)
        cells[0][0] = 'position'  # Finishing order col. has no col. name in PDF
        return pd.DataFrame(cells[1:], columns=cells[0])

    def _parse_classification(self) -> pd.DataFrame:
        """Parse "Race/Sprint Race Final Classification" PDF

        The output dataframe has columns [driver No., laps completed, total time,
        finishing position, finishing status, fastest lap time, fastest lap speed, fastest lap No.]
        """
        # Find the page with "Final Classification", on which the table is located
        doc = pymupdf.open(self.classification_file)
        found = []
        for i in range(len(doc)):
            page = Page(doc[i])
            found = page.search_for('Final Classification')
            if found:
                break
            found = page.search_for('Provisional Classification')
            if found:
                warnings.warn('Found and using provisional classification, not the final one')
                break
            else:
                found = page.get_image_header()
                if found:
                    found = [found]
                    warnings.warn('Found an image header, instead of strings')
                    break
        if not found:
            doc.close()
            raise ValueError(f'"Final Classification" or "Provisional Classification" not found '
                             f'on any page in {self.classification_file}')

        # Page width. This is the rightmost x-coord. of the table
        w = page.bound()[2]

        # Position of "Final Classification". Topmost y-coord. of the table
        y = found[0].y1

        # Bottommost y-coord. of the table, identified by "NOT CLASSIFIED" or "FASTEST LAP",
        # whichever comes first
        has_not_classified = False
        bottom = page.search_for('NOT CLASSIFIED')
        if bottom:
            has_not_classified = True
        else:
            bottom = page.search_for('FASTEST LAP')
        if not bottom:
            raise ValueError(f'Could not find "NOT CLASSIFIED" or "FASTEST LAP" in '
                             f'{self.classification_file}')
        b = bottom[0].y0

        # Table bounding box
        bbox = pymupdf.Rect(0, y, w, b)

        # Left and right x-coords. of table cols.
        pos = {}
        for col in ['NO', 'DRIVER', 'NAT', 'ENTRANT', 'LAPS', 'TIME', 'GAP', 'INT', 'KM/H',
                    'FASTEST', 'ON', 'PTS']:
            pos[col] = {
                'left': page.search_for(col, clip=bbox)[0].x0,
                'right': page.search_for(col, clip=bbox)[0].x1
            }

        # Vertical lines separating the columns
        vlines = [
            0,
            pos['NO']['left'],
            (pos['NO']['right'] + pos['DRIVER']['left']) / 2,
            pos['NAT']['left'] - 1,
            (pos['NAT']['right'] + pos['ENTRANT']['left']) / 2,
            pos['LAPS']['left'],
            pos['LAPS']['right'],
            (pos['TIME']['right'] + pos['GAP']['left']) / 2,
            (pos['GAP']['right'] + pos['INT']['left']) / 2,
            (pos['INT']['right'] + pos['KM/H']['left']) / 2,
            pos['FASTEST']['left'],
            pos['FASTEST']['right'],
            pos['PTS']['left'],
            pos['PTS']['right']
        ]

        # Horizontal lines separating the rows
        car_nos = page.get_text('blocks', clip=(pos['NO']['left'], y, pos['NO']['right'], b))
        hlines = [y]
        for i in range(len(car_nos) - 1):
            hlines.append((car_nos[i][3] + car_nos[i + 1][1]) / 2)
        hlines.append(b)

        # Parse the table using the grid above
        df = self._parse_table_by_grid(page, vlines, hlines)
        assert df.shape[1] == 13, \
            f'Expected 13 columns, got {df.shape[1]} in {self.classification_file}'

        # Do the same for the "NOT CLASSIFIED" table. See `QualifyingParser._parse_classification`
        if has_not_classified:
            t = page.search_for('NOT CLASSIFIED')[0].y1
            line_height = max([i[3] - i[1] for i in car_nos])
            car_nos = []
            while car_no := page.get_text('blocks',
                                          clip=(vlines[1], t, vlines[2], t + line_height)):
                assert len(car_no) == 1, f'Error in detecting rows in the "NOT CLASSIFIED" ' \
                                         f'table in {self.classification_file}'
                car_no = car_no[0]
                car_nos.append([car_no[1], car_no[3]])
                t = car_no[3]
            hlines = [car_nos[0][0] - 1]
            for i in range(len(car_nos) - 1):
                hlines.append((car_nos[i][1] + car_nos[i + 1][0]) / 2)
            hlines.append(car_nos[-1][1] + 1)
            not_classified = self._parse_table_by_grid(page, vlines, hlines)
            not_classified.loc[-1] = not_classified.columns
            not_classified.iloc[-1, 0] = ''  # Drop the "position" col. name
            not_classified = not_classified.sort_index().reset_index(drop=True)
            assert not_classified.shape[1] == 13, \
                f'Expected 13 columns for "NOT CLASSIFIED" table , got ' \
                f'{not_classified.shape[1]} in {self.classification_file}'
            not_classified.columns = df.columns

        else:
            # no unclassified drivers
            not_classified = pd.DataFrame(columns=df.columns)

        df['is_classified'] = True # Set all drivers from the main table as classified

        not_classified['finishing_status'] = 11  # TODO: should clean up the code later
        not_classified['is_classified'] = False

        df = pd.concat([df, not_classified], ignore_index=True)

        # Set col. names
        del df['NAT']
        df = df.rename(columns={
            'position': 'finishing_position',
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

        # Clean up finishing status, e.g. is lapped? Is DSQ?
        df.loc[df.gap.fillna('').str.contains('LAP', regex=False), 'finishing_status'] = 1
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
        df.fillna({
            'points': 0,
            'finishing_status': 0
        }, inplace=True)
        df.finishing_status = df.finishing_status.astype(int)

        # Merge in starting grid from lap chart PDF
        self._parse_lap_chart()
        df = df.merge(self.starting_grid, on='car_no', how='left')

        def to_json() -> list[dict]:
            return df.apply(
                lambda x: SessionEntryImport(
                    object_type="SessionEntry",
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

        def to_pkl(filename: str | os.PathLike) -> None:
            with open(filename, 'wb') as f:
                pickle.dump(df.to_json(), f)
            return

        df.to_json = to_json
        df.to_pkl = to_pkl
        return df

    def _parse_history_chart(self) -> pd.DataFrame:
        doc = pymupdf.open(self.history_chart_file)
        df = []
        for page in doc:
            # Each page can have multiple tables, all of which begins from the same top y-position.
            # Their table headers are vertically bounded between "History Chart" and "TIME". Find
            # all the headers
            t = page.search_for('History Chart')[0].y1
            b = page.search_for('TIME')[0].y1
            w = page.bound()[2]
            headers = page.search_for('Lap', clip=(0, t, w, b))

            # Iterate over each table header and get the table content
            h = page.bound()[3]
            for i, lap in enumerate(headers):
                """
                The left boundary of the table is the leftmost of the "Lap xxx" text, and the right
                boundary is the leftmost of the next "Lap x" text. If it's the last lap, i.e. no
                next table, then the right boundary can be determined by left boundary plus table
                width, which is roughly one-fifth of the page width. We add 5% extra buffer to the
                right boundary

                TODO: a better way to determine table width is by using the vector graphics: under
                the table header, we have a black horizontal line, whose length is the table width
                """
                left_boundary = lap.x0
                if i + 1 < len(headers):
                    right_boundary = headers[i + 1].x0
                else:
                    right_boundary = (left_boundary + w / 5) * 1.05
                temp = page.find_tables(clip=(left_boundary, t, right_boundary, h),
                                        strategy='lines',
                                        add_lines=[((left_boundary, 0), (left_boundary, h))])
                assert len(temp.tables) == 1, \
                    f'Expected one table per lap, got {len(temp.tables)} on p.{page.number} in ' \
                    f'{self.history_chart_file}'
                temp = temp[0].to_pandas()

                # Three columns: "LAP x", "GAP", "TIME". "LAP x" is the column for driver No. So
                # add a new column for lap No. with value "x", and rename the columns
                lap_no = int(temp.columns[0].split(' ')[1])
                temp.columns = ['car_no', 'gap', 'time']
                temp['lap'] = lap_no
                temp = temp[temp.car_no != '']  # Sometimes will get one additional empty row

                # The row order/index is meaningful: it's the order/positions of the cars
                # Need extra care for lapped cars; see harningle/fia-doc#19
                # TODO: is this true for all cases? E.g. retirements?
                temp.reset_index(drop=False, names=['position'], inplace=True)
                temp['position'] += 1  # 1-indexed
                df.append(temp)

        df = pd.concat(df, ignore_index=True)
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
        df.reset_index(drop=False, inplace=True)
        df.sort_values(by=['car_no', 'lap', 'index'], inplace=True)
        df.loc[(df.car_no == df.car_no.shift(-1)) & (df.lap == df.lap.shift(-1)), 'lap'] -= 1
        df.loc[(df.car_no == df.car_no.shift(1)) & (df.lap == df.lap.shift(1) + 2), 'lap'] -= 1
        del df['index']

        # TODO: Perez "retired and rejoined" in 2023 Japanese... Maybe just mechanically assign lap
        #       No. as 1, 2, 3, ... for each driver?
        return df

    def _parse_lap_chart(self) -> pd.DataFrame:
        doc = pymupdf.open(self.lap_chart_file)
        df = []
        for page in doc:
            t = page.search_for('Lap Chart')[0].y1  # The table is below "Race Lap Chart"
            b = page.search_for('page')
            if b:  # The table is above "page x of y"
                b = b[0].y0
            else:  # Or the bottom of the page (the © logo)
                b = page.search_for('©')
                if b:
                    b = b[0].y0
                else:
                    raise ValueError(f'Cannot find © on p. {page.number} in {self.lap_chart_file}')

            # Left boundary is the leftmost "LAP x"
            l = page.search_for('LAP', clip=(0, t, page.bound()[2], b))[0].x0

            # Top row's leftmost cell is "POS"
            pos = page.search_for('POS', clip=(l, t, page.bound()[2], b))[0]
            assert pos.x0 >= l, \
                f'Expected "POS" to be (slightly) to the right of "LAP x" on page {page.number} ' \
                f"in {self.lap_chart_file}. But it's found to the left"
            assert pos.y0 > t, \
                f'Expected "POS" below "Race Lap Chart" on page {page.number} in ' \
                f'{self.lap_chart_file}. But it\'s found above'

            # To the right of "POS", we have positions 1, 2, ...
            positions = page.get_text('dict', clip=(pos.x1, pos.y0, page.bound()[2], pos.y1))
            assert len(positions['blocks']) == 1, f'Error in parsing positions in top row on ' \
                                        f'page {page.number} in {self.lap_chart_file}'
            positions = positions['blocks'][0]['lines']
            assert positions, \
                f'Expected some positions to the right of "POS" on page {page.number} in ' \
                f'{self.lap_chart_file}. Found none'

            # Each position is a col., so we take the midpoints between two positions as the col.
            # separators
            col_seps = [l - 5, (pos.x1 + positions[0]['bbox'][0]) / 2]
            for i in range(len(positions) - 1):
                col_seps.append((positions[i]['bbox'][2] + positions[i + 1]['bbox'][0]) / 2)
            col_seps.append(page.bound()[2])

            # Below "POS" we have rows GRID, LAP 1, LAP 2, ...
            laps = page.get_text('dict', clip=(l - 5, pos.y1, col_seps[1], b))
            assert len(laps['blocks']) == 1, f'Error in parsing laps in leftmost col. on ' \
                                             f'p.{page.number} in {self.lap_chart_file}'
            laps = laps['blocks'][0]['lines']
            assert laps, \
                f'Expected some laps below "POS" on page {page.number} in ' \
                f'{self.lap_chart_file}. Found none'

            # Each lap is a row, so we take the midpoints between two laps as the row separators
            row_seps = [pos.y0 - 1, (pos.y1 + laps[0]['bbox'][1]) / 2]
            for i in range(len(laps) - 1):
                row_seps.append((laps[i]['bbox'][3] + laps[i + 1]['bbox'][1]) / 2)
            row_seps.append(b)

            # Parse the table
            page = Page(page)
            tab, superscript, cross_out = page.parse_table_by_grid(
                vlines=[(col_seps[i], col_seps[i + 1]) for i in range(len(col_seps) - 1)],
                hlines=[(row_seps[i], row_seps[i + 1]) for i in range(len(row_seps) - 1)]
            )
            assert (len(superscript) == 0) and len(cross_out) == 0, \
                f'Some superscript(s) or crossed out text(s) found in table on page ' \
                f'{page.number} in {self.lap_chart_file}. Expect none'
            assert (len(superscript) == 0) and len(cross_out) == 0, \
                f'Some superscript(s) or crossed out text(s) found in table on page ' \
                f'{page.number} in {self.lap_chart_file}. Expect none'

            # Reshape to long format, where a row is (lap, driver, position)
            tab.columns = tab.iloc[0]
            tab = tab[1:]
            tab.index.name = None
            if (tab.POS == 'GRID').any():
                self.starting_grid = tab[tab.POS == 'GRID'] \
                    .drop(columns='POS') \
                    .T \
                    .reset_index() \
                    .rename(columns={0: 'starting_grid', 1: 'car_no'})
                self.starting_grid.car_no = self.starting_grid.car_no.astype(int)
                self.starting_grid.starting_grid = self.starting_grid.starting_grid.astype(int)
            tab = tab[tab.POS != 'GRID']
            tab.POS = tab.POS.str.removeprefix('LAP ').astype(int)
            tab = tab.set_index('POS').stack().reset_index(name='car_no')
            tab.rename(columns={'POS': 'lap', 0: 'position'}, inplace=True)
            tab = tab[tab.car_no != '']
            tab.position = tab.position.astype(int)
            tab.car_no = tab.car_no.astype(int)
            df.append(tab)
        return pd.concat(df, ignore_index=True)

    def _parse_lap_analysis(self) -> pd.DataFrame:
        doc = pymupdf.open(self.lap_analysis_file)
        b, r = doc[0].bound()[3], doc[0].bound()[2]
        df = []
        for page in doc:
            # Get the position of "LAP" and "TIME" on the page. Can have multiple of them. The
            # tables are below these texts
            h = page.search_for('Lap Analysis')[0].y1
            laps = page.search_for('LAP ', clip=(0, h, r, b))
            """
            "LAP" can have false positive match, if a driver's name contains "LAP", e.g. Colapinto.
            So we add a whitespace after "LAP" to avoid this. For the same reason, we also add a
            whitespace after "TIME" below.
            """
            # TODO: check
            times = page.search_for('TIME ', clip=(0, h, r, b))
            assert len(laps) == len(times), \
                f'#. of "LAP" and #. of "TIME" do not match on p.{page.number} in ' \
                f'{self.lap_analysis_file}'

            # If it's race, then only three drivers (or less) on one page. And all of them should
            # start from the same y-coord. and have the same y-coord. for "LAP" and "TIME"
            if self.session == 'race':
                assert len(laps) <= 6, f'Expected at most 6 "LAP" on p.{page.number} in ' \
                                       f'{self.lap_analysis_file}. Found {len(laps)}'
                for i in range(len(laps) - 1):
                    assert np.isclose(laps[i].y1, laps[i + 1].y1, atol=1) and \
                              np.isclose(times[i].y1, times[i + 1].y1, atol=1), \
                        f'y-coord. of "LAP" and "TIME" do not match on p.{page.number} in ' \
                        f'{self.lap_analysis_file}: {laps[i].y1} vs {laps[i + 1].y1} vs ' \
                        f'{times[i].y1} vs {times[i + 1].y1}'
            # If it's sprint, then can have many more drivers in total. However, in one row, it's
            # still three drivers side by side, and the last row may have less, and in each row,
            # "LAP" and "TIME" should have the same y-coord.
            else:
                laps.sort(key=lambda x: x.y0)
                times.sort(key=lambda x: x.y0)
                for i in range(0, len(laps) - 1, 6):
                    for j in range(6):
                        if i + j == len(laps):
                            break
                        assert np.isclose(laps[i + j].y0, laps[i].y0, atol=1) and \
                                np.isclose(times[i + j].y0, times[i].y0, atol=1), \
                            f'y-coord. of "LAP" and "TIME" do not match in row {i} on ' \
                            f'p.{page.number} in {self.lap_analysis_file}: {laps[i + j].y0} vs ' \
                            f'{laps[i].y0} vs {times[i + j].y0} vs {times[i].y0}'

            # Horizontally, three drivers share the full width of the page, side by side. If it's
            # race, then three drivers (or less) on one page
            # See QualifyingParser._parse_lap_times() for detailed explanation
            if self.session == 'race':
                w = r / 3
                for i in range(3):
                    # Driver's name is below "Race Lap Analysis" and above the table
                    l = i * w
                    r = (i + 1) * w
                    t = min([i.y0 for i in laps] + [i.y0 for i in times])
                    driver = page.get_text('block', clip=(l, h, r, t)).strip()
                    if not driver:
                        continue
                        # TODO: may want a test here. Every row above should have exactly 3 drivers
                    car_no, driver = driver.split('\n', 1)

                    # Each driver has two tables side by side. We parse the tables by manually
                    # specifying row and col. positions (harningle/fia-doc#17)
                    # TODO: need to check if always have two tables. A good edge case is Spa 2021
                    #       or Leclerc DNS in 2023 Brazil

                    # Find the horizontal lines under which the tables are located
                    page = Page(page)
                    t = max([i.y1 for i in laps] + [i.y1 for i in times])
                    lines = [i for i in page.get_drawings_in_bbox((l, t, r, t + 10))
                             if np.isclose(i['rect'].y0, i['rect'].y1, atol=1)]
                    assert len(lines) >= 1, f'Expected at least one horizontal line for ' \
                                            f'table(s) in col. {i} in page {page.number} in ' \
                                            f'{self.history_chart_file}. Found none'
                    assert np.allclose([i['rect'].y0 for i in lines],
                                       lines[0]['rect'].y0,
                                       atol=1), \
                        f'Horizontal lines for table(s) in col. {i} in page {page.number} in ' \
                        f'{self.history_chart_file} are not at the same y-position'

                    # Concat lines.
                    lines.sort(key=lambda x: x['rect'].x0)
                    rect = lines[0]['rect']
                    top_lines = [(rect.x0, rect.y0, rect.x1, rect.y1)]
                    for line in lines[1:]:
                        rect = line['rect']
                        prev_line = top_lines[-1]
                        # If one line ends where the other starts, they are the same line
                        if np.isclose(rect.x0, prev_line[2], atol=1):
                            top_lines[-1] = (prev_line[0], prev_line[1], rect.x1, prev_line[3])
                        # If one line starts where the other ends, they are the same line
                        elif np.isclose(rect.x1, prev_line[0], atol=1):
                            top_lines[-1] = (rect.x0, prev_line[1], prev_line[2], prev_line[3])
                        # Otherwise, it's a new line
                        else:
                            top_lines.append((rect.x0, rect.y0, rect.x1, rect.y1))
                    assert len(top_lines) in [1, 2], \
                        f'Expected at most two horizontal lines for table(s) in col. {i} in ' \
                        f'p.{page.number} in {self.history_chart_file}. Found {len(top_lines)}'

                    # Find the column separators
                    col_seps = []
                    for line in top_lines:
                        no = page.search_for('LAP', clip=(line[0], line[1] - 15, line[2], line[3]))
                        assert len(no) == 1, \
                            f'Expected exactly one "LAP" above the top line at ' \
                            f'({line[0], line[1], line[2], line[3]}) in page {page.number} in ' \
                            f'{self.history_chart_file}. Found {len(no)}'
                        col_seps.append([
                            (line[0], no[0].x1),
                            (no[0].x1, (line[0] + line[2]) / 2 - 5),
                            ((line[0] + line[2]) / 2 - 5, line[2])
                        ])

                    # Find the white and grey rectangles under the top lines. Each row is either
                    # coloured/filled in either white or grey, so we can get the row's top and
                    # bottom y-positions from these rectangles
                    rects = [i for i in page.get_drawings_in_bbox((l, t, r, b))
                             if i['rect'].y1 - i['rect'].y0 > 10]
                    ys = [j for i in rects for j in [i['rect'].y0, i['rect'].y1]]
                    ys.sort()
                    row_seps = [ys[0]]
                    for y in ys[1:]:
                        if y - row_seps[-1] > 10:
                            row_seps.append(y)
                    row_seps = [(row_seps[i], row_seps[i + 1]) for i in range(len(row_seps) - 1)]

                    # Finally we are good to parse the tables using these separators
                    temp = []
                    for cols in col_seps:
                        tab, superscript, cross_out = page.parse_table_by_grid(
                            vlines=cols, hlines=row_seps
                        )
                        assert (len(superscript) == 0) and len(cross_out) == 0, \
                            f'Some superscript(s) or crossed out text(s) found in table at ' \
                            f'({cols[0][0]:.1f}, {row_seps[0][0]:.1f}, {cols[2][1]:.1f}, ' \
                            f'{row_seps[-1][1]:.1f}) in page {page.number} in ' \
                            f'{self.history_chart_file}. But we expect none'
                        assert tab.shape[1] == 3, \
                            f'Expected three columns (LAP, pit or not, lap time) in table at ' \
                            f'({cols[0][0]:.1f}, {row_seps[0][0]:.1f}, {cols[2][1]:.1f}, ' \
                            f'{row_seps[-1][1]:.1f}) in page {page.number} in ' \
                            f'{self.history_chart_file}. Found {len(tab)}'
                        tab.columns = ['lap', 'pit', 'lap_time']

                        # Drop empty row
                        """
                        This is because the two side-by-side tables may not have the same amount of
                        rows. E.g., there are 11 laps, and the left table will have 6 and the right
                        table has 5 rows. The right table will have an empty row at the bottom, so
                        drop it here
                        """
                        tab = tab[tab.lap != '']
                        temp.append(tab)

                    # One driver may have multiple tables. Concatenate them
                    temp = pd.concat(temp, ignore_index=True)
                    if temp.empty:
                        warnings.warn(f'Driver {driver}, car No. {car_no}, has no lap at all on '
                                      f'page {page.number} in {self.lap_analysis_file}. Make sure '
                                      f'this is expected, e.g. DNS')
                    temp['car_no'] = car_no
                    temp['driver'] = driver
                    df.append(temp)
            # If it's sprint, then can have many rows on one page, and each row has three drivers
            # side by side
            else:
                # y-coords. of "LAP" and "TIME". They are the top of each table
                page = Page(page)
                ys = [i.y1 for i in laps]
                ys.sort()  # Sort these "LAP"'s from top to bottom
                top_pos = [ys[0]]
                for y in ys[1:]:
                    if y - top_pos[-1] > 10:
                        top_pos.append(y)

                # Bottom of the table is the next "NO TIME", or the bottom of the page
                ys = [i.y0 for i in laps]
                ys.sort()
                bottom_pos = [ys[0]]
                for y in ys[1:]:
                    if y - bottom_pos[-1] > 10:
                        bottom_pos.append(y)
                b = page.bound()[3]
                bottom_pos.append(b)
                bottom_pos = bottom_pos[1:]  # The first "NO TIME" is not the bottom of any table

                # Find the tables located between each `top_pos` and `bottom_pos`
                w = page.bound()[2]
                for row in range(len(top_pos)):
                    # Each row usually has three drivers. Iterate over drivers
                    for col in range(3):

                        # Find the driver name, which is located immediately above the table
                        driver = page.get_text(
                            'block',
                            clip=(
                                col * w / 3,        # Each driver occupies ~1/3 of the page width
                                top_pos[row] - 30,  # Driver name is ~20-30 px above the table
                                (col + 1) * w / 3,
                                top_pos[row] - 10
                            )
                        ).strip()
                        if not driver:
                            continue
                            # TODO: may want a test here. Every row above should have
                            #       precisely three drivers
                        car_no, driver = driver.split(maxsplit=1)

                        # Find the horizontal line(s) below "NO" and "TIME". This is the top of the
                        # table(s)
                        bbox = (col * w / 3, top_pos[row], (col + 1) * w / 3, bottom_pos[row])
                        lines = [i for i in page.get_drawings_in_bbox(bbox)
                                 if np.isclose(i['rect'].y0, i['rect'].y1, atol=1)
                                 and i['fill'] is None]
                        assert len(lines) >= 1, \
                            f'Expected at least one horizontal line for table(s) in row {row}, ' \
                            f'col {col} in p.{page.number} in {self.lap_analysis_file}. Found none'
                        assert np.allclose(
                            [i['rect'].y0 for i in lines],
                            lines[0]['rect'].y0,
                            atol=1
                        ), \
                            f'Horizontal lines for table(s) in row {row}, col {col} on p.' \
                            f'{page.number} in {self.lap_analysis_file} are not at the same '\
                            f'y-coord.'

                        # Concat lines.
                        lines.sort(key=lambda x: x['rect'].x0)
                        rect = lines[0]['rect']
                        top_lines = [(rect.x0, rect.y0, rect.x1, rect.y1)]
                        for line in lines[1:]:
                            rect = line['rect']
                            prev_line = top_lines[-1]
                            # If one line ends where the other starts, they are the same line
                            if np.isclose(rect.x0, prev_line[2], atol=1):
                                top_lines[-1] = (prev_line[0], prev_line[1], rect.x1, prev_line[3])
                            # If one line starts where the other ends, they are the same line
                            elif np.isclose(rect.x1, prev_line[0], atol=1):
                                top_lines[-1] = (rect.x0, prev_line[1], prev_line[2], prev_line[3])
                            # Otherwise, it's a new line
                            else:
                                top_lines.append((rect.x0, rect.y0, rect.x1, rect.y1))
                        assert len(top_lines) in [1, 2], \
                            f'Expected at most two horizontal lines for table(s) in row {row}, ' \
                            f'col. {col} on p.{page.number} in {self.lap_analysis_file}. Found ' \
                            f'{len(top_lines)}'

                        # Find the column separators
                        col_seps = []
                        for line in top_lines:
                            lap = page.search_for('LAP',
                                                  clip=(line[0], line[1] - 15, line[2], line[3]))
                            assert len(lap) == 1, \
                                f'Expected exactly one "LAP" above the top line at (' \
                                f'{line[0], line[1], line[2], line[3]}) on p.{page.number} in ' \
                                f'{self.lap_analysis_file}. Found {len(lap)}'
                            col_seps.append([
                                (line[0], lap[0].x1),
                                (lap[0].x1, (line[0] + line[2]) / 2 - 5),
                                ((line[0] + line[2]) / 2 - 5, line[2])
                            ])

                        # Find the white and grey rectangles under the top lines. Each row is either
                        # coloured/filled in white or grey, so we can get the row's top and bottom
                        # y-positions from these rectangles
                        rects = [i for i in page.get_drawings_in_bbox(bbox)
                                 if i['rect'].y1 - i['rect'].y0 > 10]
                        ys = [j for i in rects for j in [i['rect'].y0, i['rect'].y1]]
                        ys.sort()
                        row_seps = [ys[0]]
                        for y in ys[1:]:
                            if y - row_seps[-1] > 10:
                                row_seps.append(y)
                        row_seps = [(row_seps[i], row_seps[i + 1])
                                    for i in range(len(row_seps) - 1)]

                        # Finally we are good to parse the tables using these separators
                        temp = []
                        for cols in col_seps:
                            tab, superscript, cross_out = page.parse_table_by_grid(
                                vlines=cols, hlines=row_seps
                            )
                            assert (len(superscript) == 0) and (len(cross_out) == 0), \
                                f'Some superscript(s) or cross-out texts found in table at ' \
                                f'({cols[0][0]:.1f}, {row_seps[0][0]:.1f}, {cols[2][1]:.1f}, ' \
                                f'{row_seps[-1][1]:.1f}) on p.{page.number} in ' \
                                f'{self.lap_analysis_file}. But expect none'
                            # Drop empty row
                            tab = tab[tab[0] != '']
                            temp.append(tab)

                        # One driver may have multiple tables. Concatenate them
                        temp = pd.concat(temp, ignore_index=True)
                        temp['car_no'] = car_no
                        temp['driver'] = driver
                        temp.rename(columns={0: 'lap', 1: 'pit', 2: 'lap_time'}, inplace=True)
                        df.append(temp)

        df = pd.concat(df, ignore_index=True)
        df.lap = df.lap.astype(int)
        df.pit = df.pit.apply(lambda x: x == 'P')
        df.car_no = df.car_no.astype(int)
        return df

    def _parse_lap_times(self) -> pd.DataFrame:
        # Get lap times from Race Lap Analysis PDF
        df = self._parse_lap_analysis()

        # Lap 1's lap times are calendar time in Race Lap Analysis. To get the actual lap time for
        # lap 2, we parse Race History Chart PDF
        lap_1 = self._parse_history_chart()
        lap_1 = lap_1[lap_1.lap == 1][['car_no', 'lap', 'time']]
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
        assert (temp.lap_time == temp.fastest_lap_time).all(), \
            'Fastest lap time in lap times does not match the one in final classification'
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

        def to_pkl(filename: str | os.PathLike) -> None:
            with open(filename, 'wb') as f:
                pickle.dump(df.to_json(), f)

        df.to_json = to_json
        df.to_pkl = to_pkl
        return df

    def _cross_validate(self) -> None:
        """Cross validate against other PDFs or fastf1?"""
        raise NotImplementedError


class QualifyingParser:
    """
    TODO: need better docstring
    TODO: probably need to refactor this. Not clean
    Quali. sessions have to be parsed using multiple PDFs jointly. Otherwise, we don't know which
    lap is in which quali. session
    """
    def __init__(
            self,
            classification_file: str | os.PathLike,
            lap_times_file: str | os.PathLike,
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
        self.classification_df = self._parse_classification()
        self.lap_times_df = self._parse_lap_times()
        # self._cross_validate()

    def _check_session(self) -> None:
        """Check that the input session is valid. Raise an error otherwise"""
        if self.session not in get_args(QualiSessionT):
            raise ValueError(f'Invalid session: {self.session}. '
                             f'Valid sessions are: {get_args(QualiSessionT)}"')
        # TODO: 2023 US sprint shootout. No "POLE POSITION LAP"???
        return

    def _parse_table_by_grid(
            self,
            page: Page,
            vlines: list[float],
            hlines: list[float],
            tol: float = 2
    ) -> pd.DataFrame:
        """Manually parse the table cell by cell, defined by lines separating the columns and rows

        See `EntryListParser._parse_table_by_grid()` for detailed explanation.

        :param vlines: x-coords. of vertical lines separating the cols. Table left and right
                       boundaries need to be included
        :param hlines: y-coords. of horizontal lines separating the rows. Table top and bottom
                       boundaries need to be included
        :param tol: tolerance for bbox. of text. Default is 2 pixels
        """
        cells = []
        for i in range(len(hlines) - 1):
            row = []
            for j in range(len(vlines) - 1):
                text = ''
                l, t, r, b = vlines[j], hlines[i], vlines[j + 1], hlines[i + 1]
                cell = page.get_text('blocks', clip=(l, t, r, b))
                if cell:
                    # Usually, one cell is one line of text. The only exception is Andrea Kimi
                    # Antonelli. His name is too long and thus wrapped into two lines
                    if len(cell) > 1:
                        if len(cell) == 2 and cell[0][4].strip() == 'Andrea Kimi':
                            text = cell[0][4].strip() + ' ' + cell[1][4].strip()
                        else:
                            raise Exception(f'Expected exactly one cell in row {i}, col {j} in '
                                            f'{self.classification_file}. Found {cell}')
                    cell = cell[0]
                    if cell[4].strip():
                        bbox = cell[0:4]
                        if bbox[0] < vlines[j] - tol or bbox[2] > vlines[j + 1] + tol \
                                or bbox[1] < t - tol or bbox[3] > b + tol:
                            raise ValueError(f'Found text outside the cell in row {i}, col {j} in '
                                             f'{self.classification_file}: {cell}')
                        text = cell[4].strip()
                row.append(text)
            cells.append(row)
        cells[0][0] = 'position'  # Finishing order col. has no col. name in PDF
        return pd.DataFrame(cells[1:], columns=cells[0])

    def _parse_classification(self):
        # Find the page with "Qualifying Session Final Classification"
        doc = pymupdf.open(self.classification_file)
        found = []
        for i in range(len(doc)):
            page = Page(doc[i])
            found = page.search_for('Final Classification')
            if found:
                break
            found = page.search_for('Provisional Classification')
            if found:
                warnings.warn('Found and using provisional classification, not the final one')
                break
            else:
                found = page.get_image_header()
                if found:
                    found = [found]
                    warnings.warn('Found an image header, instead of strings')
                    break
        if not found:
            doc.close()  # TODO: check docs. Do we need to manually close it? Memory safe?
            raise ValueError(f'"Final Classification" or "Provisional Classification" not found '
                             f'on any page in {self.classification_file}')

        # Page width. This is the rightmost x-coord. of the table
        w = page.bound()[2]

        # y-position of "Final Classification", which is the topmost y-coord. of the table
        y = found[0].y1

        # y-position of "NOT CLASSIFIED - " or "POLE POSITION LAP", whichever comes the first. This
        # is the bottom of the classification table. Some PDFs may not have these texts. In these
        # cases, we use the long black thick horizontal line to determine the bottom of the table
        has_not_classified = False
        bottom = page.search_for('NOT CLASSIFIED - ')
        if bottom:
            has_not_classified = True
        else:
            bottom = page.search_for('POLE POSITION LAP')
        if not bottom:
            lines = page.get_drawings_in_bbox((0, y + 50, w, page.bound()[3]))
            lines = [i for i in lines
                     if np.isclose(i['rect'].y0, i['rect'].y1, atol=1)
                     and i['width'] is not None
                     and np.isclose(i['width'], 1, rtol=0.1)
                     and i['rect'].x1 - i['rect'].x0 > 0.8 * w]
            if not lines:
                # Go though the pixel map and find a wide horizontal white strip with 10+ px height
                pixmap = page.get_pixmap(clip=(0, y + 50, w, page.bound()[3]))
                l, t, r, b = pixmap.x, pixmap.y, pixmap.x + pixmap.w, pixmap.y + pixmap.h
                pixmap = np.ndarray([b - t, r - l, 3], dtype=np.uint8, buffer=pixmap.samples_mv)
                is_white_row = np.all(pixmap == 255, axis=(1, 2))
                white_strips = []
                strip_start = None
                for i, is_white in enumerate(is_white_row):
                    if is_white and strip_start is None:
                        strip_start = i
                    elif not is_white and strip_start is not None:
                        if i - strip_start >= 10:  # At least 10 rows of white
                            white_strips.append(strip_start + t)
                        strip_start = None
                # If the strip is at the bottom. Shouldn't happen but just in case
                if strip_start is not None and len(is_white_row) - strip_start >= 10:
                    white_strips.append(strip_start + t)
                if not white_strips:
                    raise ValueError(f'Could not find "NOT CLASSIFIED - " or "POLE POSITION LAP" '
                                     f'or a thick horizontal line in {self.classification_file}')
                strip_start = white_strips[0]  # The topmost one is the bottom of the table
                bottom = [pymupdf.Rect(l, strip_start + 1, r, strip_start + 2)]  # One pixel buffer
            else:
                lines.sort(key=lambda x: x['rect'].y0)
                bottom = [lines[0]['rect']]
        b = bottom[0].y0

        # Get the location of cols.
        """
        The default `page.find_tables` was working fine until Antonelli. His full name is long, so
        that the horizontal gap between the driver name col. and nationality col. in his row is
        narrow. This breaks the automatic col. detection of pymupdf. Therefore, we need to manually
        specify the vertical lines separating cols.
        """
        no = page.search_for('NO', clip=(0, y, w, b))
        no.sort(key=lambda x: x.y0)  # The topmost "NO" under "Final Classification"
        no = no[0]
        b_header = no.y1 + 1  # The bottom of the header row
        headers = page.get_text('text', clip=(0, y, w, b_header)).split()  # Col. headers/names
        cols: dict[str, tuple[float, float]] = {}
        l = no.x0 - 1
        q = 1
        for col in headers:
            col_name = col
            col = page.search_for(col, clip=(l, y, w, b_header))
            # These col. names are unique
            if col_name not in ['LAPS', 'TIME']:
                assert len(col) == 1, f'Expected exactly one "{col}" in the header row in ' \
                                      f'{self.classification_file}. Found {len(col)}'
            # We will have three "LAPS" and "TIME" for Q1, Q2, and Q3
            else:
                assert len(col) == 4 - q, f'Expected {4 - q} "{col}" in the header row after ' \
                                          f'Q{q} (incl.) in {self.classification_file}. Found ' \
                                          f'{len(col)}'
                col.sort(key=lambda x: x.x0)
                col_name = f'Q{q}_{col_name}'
            col = col[0]
            cols[col_name] = (col.x0, col.x1)
            # Update the new leftmost x-coord. for the next col. I.e., instead of starting from the
            # very left of the page, start from the right of the current col.
            l = col.x1
            # Update the session if necessary
            if 'Q2' in col_name:  # Captures both "Q2" and "SQ2"
                q = 2
            elif 'Q3' in col_name:
                q = 3
        for col in ['SQ1', 'SQ2', 'SQ3']:
            if col in cols:
                cols[col.replace('SQ', 'Q')] = cols.pop(col)

        # Specify the vertical lines separating the cols.
        shifter = 1.1 if self.session == 'quali' else 0.8
        """
        We don't have very good ways to detect the left and right boundary for col. "Q2" or "SQ2".
        Generally, the left of "Q2" shifted to the left by a bit will do the job. We define "a bit"
        as a fraction of the width of "Q2", so that the page/text size won't affect the detection.
        Howover, the width of "Q2" and "SQ2" may not be the same, so the shifter will be different.
        """
        vlines = [
            0,                                                        # Left of the page
            cols['NO'][0] - 1,                                        # Left of "NO"
            (cols['NO'][1] + cols['DRIVER'][0]) / 2,                  # Between "NO" and "DRIVER"
            cols['NAT'][0] - 1,                                       # Left of "NAT"
            (cols['NAT'][1] + cols['ENTRANT'][0]) / 2,                # Between "NAT" and "ENTRANT"
            (1 + shifter) * cols['Q1'][0] - shifter * cols['Q1'][1],  # See notes above
            cols['Q1_LAPS'][0],                                       # Left of "Q1_LAPS"
            cols['Q1_LAPS'][1],                                       # Right of "Q1_LAPS"
            (1 + shifter) * cols['Q2'][0] - shifter * cols['Q2'][1],
            cols['Q2_LAPS'][0],                                       # Left of "Q2_LAPS"
            cols['Q2_LAPS'][1],                                       # Right of "Q2_LAPS"
            (1 + shifter) * cols['Q3'][0] - shifter * cols['Q3'][1],
            cols['Q3_LAPS'][0],                                       # Left of "Q3_LAPS"
            cols['Q3_LAPS'][1],                                       # Right of "Q3_LAPS"
            w                                                         # Right of the page
        ]
        if '%' in headers:  # Some PDFs may have one additional col. for Q1 cutoff percentage
            vlines.insert(headers.index('%') + 2,
                          1.4 * cols['Q1_TIME'][0] - 0.4 * cols['Q1_TIME'][1])

        # Get the row positions. The rows are coloured in grey, white, grey, white, ... So we just
        # need to get the top and bottom positions of the grey rectangles
        rects = []
        for i in page.get_drawings_in_bbox(bbox=(0, b_header - 2, w, b)):
            if i['fill'] is not None and np.allclose(i['fill'], 0.9, rtol=0.05):
                rects.append(i['rect'].y0 + 1)
                rects.append(i['rect'].y1 - 1)
        rects.sort()
        hlines: list[float] = [y, b_header]
        for i in rects:
            if i - hlines[-1] > 5:
                hlines.append(i)
        if b - hlines[-1] > 5:
            hlines.append(b)

        # Get the table
        df = self._parse_table_by_grid(page, vlines, hlines)

        # Clean up column name, e.g. "TIME" -> "Q2_TIME"
        """
        We name the sessions as "Q1", "Q2", and "Q3", regardless of whether it's a normal
        qualifying or a sprint qualifying. This makes the code simpler, and we should always use
        `self.session` to determine what session it is.
        """
        cols = df.columns.tolist()
        headers = [i.replace('SQ', 'Q') if i.startswith('SQ') else i for i in cols]
        i = headers.index('Q1') + 1  # TODO: rewrite this. I myself don't understand now...
        for q in [1, 2, 3]:
            while i < len(headers) and headers[i] != f'Q{q + 1}':
                if headers[i] != f'Q{q}':
                    headers[i] = f'Q{q}_{headers[i]}'  # E.g., "TIME" --> "Q2_TIME"
                i += 1
        df.columns = headers
        df['finishing_status'] = 0
        df['original_order'] = range(1, len(df) + 1)  # Driver's original order in the PDF
        df = df[(df.position != '') & df.position.notna()]

        # Do the same for the "NOT CLASSIFIED" table
        if has_not_classified:
            # Locate the bottom of the table
            # TODO: refactor this. This is a copy of the above code
            """
            The bottom of "NOT CLASSIFIED" table is usually "POLE POSITION LAP", but some PDFs do
            not have it, e.g. 2023 Australian. For these PDFs, we use a thick horizontal line,
            which is the top of "POLE POSITION LAP" table, as the bottom of the "NOT CLASSIFIED"
            table.
            """
            bottom = page.search_for('POLE POSITION LAP')
            if not bottom:
                lines = page.get_drawings_in_bbox((0, hlines[-1], w, page.bound()[3]))
                lines = [i for i in lines
                         if np.isclose(i['rect'].y0, i['rect'].y1, atol=1)
                         and i['width'] is not None
                         and np.isclose(i['width'], 1, rtol=0.1)
                         and i['rect'].x1 - i['rect'].x0 > 0.8 * w]
                if not lines:
                    # Go through the pixel map and find a wide horizontal white strip with 10+ px height
                    pixmap = page.get_pixmap(clip=(0, y + 50, w, page.bound()[3]))
                    l, t, r, b = pixmap.x, pixmap.y, pixmap.x + pixmap.w, pixmap.y + pixmap.h
                    pixmap = np.ndarray([b - t, r - l, 3], dtype=np.uint8, buffer=pixmap.samples_mv)
                    is_white_row = np.all(pixmap == 255, axis=(1, 2))
                    white_strips = []
                    strip_start = None
                    for i, is_white in enumerate(is_white_row):
                        if is_white and strip_start is None:
                            strip_start = i
                        elif not is_white and strip_start is not None:
                            if i - strip_start >= 10:  # At least 10 rows of white
                                white_strips.append(strip_start + t)
                            strip_start = None
                    # If the strip is at the bottom. Shouldn't happen but just in case
                    if strip_start is not None and len(is_white_row) - strip_start >= 10:
                        white_strips.append(strip_start + t)
                    if not white_strips:
                        raise ValueError(f'Could not find "NOT CLASSIFIED - " or "POLE POSITION LAP" '
                                         f'or a thick horizontal line in {self.classification_file}')
                    strip_start = white_strips[0]  # The topmost one is the bottom of the table
                    bottom = [pymupdf.Rect(l, strip_start + 1, r, strip_start + 2)]  # One pixel buffer
                else:
                    lines.sort(key=lambda x: x['rect'].y0)
                    bottom = [lines[0]['rect']]
            b = bottom[0].y0

            # Use grey and white rectangles to determine the rows again
            t = page.search_for('NOT CLASSIFIED - ')[0].y1
            rects = []
            for i in page.get_drawings_in_bbox(bbox=(0, t, w, b)):
                if i['fill'] is not None and np.allclose(i['fill'], 0.9, rtol=0.05):
                    rects.append(i['rect'].y0 + 1)
                    rects.append(i['rect'].y1 - 1)
            rects.sort()
            hlines = [t + 1]
            for i in rects:
                if i - hlines[-1] > 5:
                    hlines.append(i)
            if b - hlines[-1] > 5:
                hlines.append(b)
            not_classified = self._parse_table_by_grid(page, vlines, hlines)
            # No col. header in NOT CLASSIFIED table. The detected col. header is actually table
            # content, so need to append the col. header to the table
            not_classified.loc[-1] = not_classified.columns
            not_classified = not_classified[(not_classified.position != '')
                                            & not_classified.position.notna()]
            not_classified.iloc[-1, 0] = ''  # Drop the "position" col. name
            not_classified = not_classified.sort_index().reset_index(drop=True)
            not_classified['finishing_status'] = 11  # TODO: should clean up the code later
            not_classified.columns = df.columns.drop('original_order')
            not_classified = not_classified[(not_classified.NO != '') | not_classified.NO.isnull()]
            n = len(df)
            not_classified['original_order'] = range(n + 1, n + len(not_classified) + 1)
            df = pd.concat([df, not_classified], ignore_index=True)

        # Fill in the position for DNF and DSQ drivers. See `QualifyingParser` for details
        df = df.replace({'': None})
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
        df.NO = df.NO.astype(int)
        del df['NAT']
        df = df.replace('', None)
        df.position = df.position.astype(int)

        # Overwrite `.to_json()` and `.to_pkl()` methods
        # TODO: bad practice
        def to_json() -> list[dict]:
            data = []
            for q in [1, 2, 3]:
                n_drivers = QUALI_DRIVERS[self.year][q]
                temp = df[df.original_order <= n_drivers].copy()
                # Clean up DNS/DNF/DSQ drivers
                temp.loc[temp[f'Q{q}'].isin(['DQ', 'DSQ']), 'finishing_status'] = 20
                temp.loc[temp[f'Q{q}'] == 'DNF', 'finishing_status'] = 11
                temp.loc[temp[f'Q{q}'] == 'DNS', 'finishing_status'] = 30
                temp.loc[temp[f'Q{q}'].isin(['DNS', 'DNF']), f'Q{q}'] = 'Z'
                temp['is_dsq'] = (temp.finishing_status == 20)
                """
                Drivers finishing normally, incl. DNF or DNS, are ranked first. Then DSQ drivers
                are at the bottom. For normal drivers, we order them by quali. lap time. DNF or DNS
                drivers will have lap time being replaced by "Z" above, so they will be ranked
                after normal finishing drivers. And their relative order will be the same as their
                original order in the PDF. DSQ drivers are ranked first by their lap time, if not
                missing, and if missing then by their original order in the PDF
                """
                temp = temp.sort_values(by=['is_dsq', f'Q{q}', 'original_order'])
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
                                is_classified=(x.finishing_status == 0)
                            )
                        ]
                    ).model_dump(exclude_unset=True),
                    axis=1
                )
                data.extend(temp['classification'].tolist())
            return data

        def to_pkl(filename: str | os.PathLike) -> None:
            with open(filename, 'wb') as f:
                pickle.dump(to_json(), f)
            return

        df.to_json = to_json
        df.to_pkl = to_pkl
        return df

    @staticmethod
    def _assign_session_to_lap(classification: pd.DataFrame, lap_times: pd.DataFrame) \
            -> pd.DataFrame:
        """TODO: probably need to refactor this later... To tedious now"""
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
            lap_times.fillna({f'Q{q}_TIME': lap_times[f'Q{q}_TIME_y']}, inplace=True)
            del lap_times[f'Q{q}_TIME_y']

            # Check if all drivers in the final classification are merged
            temp = classification[['NO', f'Q{q}_TIME']].merge(
                lap_times[lap_times[f'Q{q}_TIME'].notnull()][['car_no']],
                left_on='NO',
                right_on='car_no',
                indicator=True
            )
            temp.dropna(subset=f'Q{q}_TIME', inplace=True)
            assert (temp['_merge'] == 'both').all(), \
                f"Some drivers' fastest laps in Q{q} cannot be found in lap times PDF: " \
                f"{', '.join([str(i) for i in temp[temp._merge != 'both']['NO']])}"
            lap_times.loc[lap_times[f'Q{q}_TIME'].notnull(), 'is_fastest_lap'] = True
            del lap_times[f'Q{q}_TIME']
        return lap_times

    def _parse_lap_times(self) -> pd.DataFrame:
        """Parse "Qualifying/Sprint Quali./Shootout Session Lap Times" PDF"""
        doc = pymupdf.open(self.lap_times_file)
        df = []
        for page in doc:
            # Page width
            page = Page(page)
            w = page.bound()[2]

            # Positions of "NO" and "TIME". They are the top of each table. One driver may have
            # multiple tables starting from roughly the same top y-position
            no_time_pos = page.search_for('NO TIME')
            assert len(no_time_pos) >= 1, \
                f'Expected at least one "NO TIME", got {len(no_time_pos)} in {self.lap_times_file}'
            ys = [i.y1 for i in no_time_pos]
            ys.sort()  # Sort these "NO"'s from top to bottom
            top_pos = [ys[0]]
            for y in ys[1:]:
                # Many "NO"'s are roughly at the same height (usually three drivers share the full
                # width of the page, and each of them have two tables side by side, so six tables
                # and six "NO"'s are vertically at the same y-position). We only need those at
                # different/unique y-positions. If there is a 10+ px vertical gap, we take it as a
                # "NO" at a lower y-position
                if y - top_pos[-1] > 10:
                    top_pos.append(y)

            # Bottom of the table is the next "NO TIME", or the bottom of the page
            ys = [i.y0 for i in no_time_pos]
            ys.sort()
            bottom_pos = [ys[0]]
            for y in ys[1:]:
                if y - bottom_pos[-1] > 10:
                    bottom_pos.append(y)
            b = page.bound()[3]
            bottom_pos.append(b)
            bottom_pos = bottom_pos[1:]  # The first "NO TIME" is not the bottom of any table

            # Find the tables located between each `top_pos` and `bottom_pos`
            for row in range(len(top_pos)):
                # Each row usually has three drivers. Iterate over each driver
                for col in range(3):

                    # Find the driver name, which is located immediately above the table
                    driver = page.get_text(
                        'block',
                        clip=(
                            col * w / 3,        # Each driver occupies ~1/3 of the page width
                            top_pos[row] - 30,  # Driver name is usually 20-30 px above the table
                            (col + 1) * w / 3,
                            top_pos[row] - 10
                        )
                    ).strip()
                    if not driver:  # In the very last row may not have all three drivers. E.g., 20
                        continue    # drivers, 3 per row, so the last row only has 2 drivers
                                    # TODO: may want a test here. Every row above should have
                                    #       precisely three drivers
                    car_no, driver = driver.split(maxsplit=1)

                    # Find the horizontal line(s) below "NO" and "TIME". This is the top of the
                    # table(s)
                    bbox = (col * w / 3, top_pos[row], (col + 1) * w / 3, bottom_pos[row])
                    lines = [i for i in page.get_drawings_in_bbox(bbox)
                             if np.isclose(i['rect'].y0, i['rect'].y1, atol=1)
                             and i['fill'] is None]
                    """
                    The horizontal lines inside the table area may not always be the line below its
                    header. When we have lap time deleted, we will have additional lines with grey
                    colour. The `i['fill'] is None` filters out these lap time deleted lines
                    """
                    assert len(lines) >= 1, f'Expected at least one horizontal line for ' \
                        f'table(s) in row {row}, col {col} in page {page.number} in ' \
                        f'{self.lap_times_file}. Found none'
                    assert np.allclose(
                        [i['rect'].y0 for i in lines],
                        lines[0]['rect'].y0,
                        atol=1
                    ), \
                        f'Horizontal lines for table(s) in row {row}, col {col} in page ' \
                        f'{page.number} in {self.lap_times_file} are not at the same y-position'

                    # Concat lines.
                    """
                    The lines above are can be segmented. E.g., one line is from x = 0 to x = 100,
                    and another is from x = 101 to x = 200. The two lines are basically one line,
                    so we want to horizontally concatenate them
                    """
                    lines.sort(key=lambda x: x['rect'].x0)
                    rect = lines[0]['rect']
                    top_lines = [(rect.x0, rect.y0, rect.x1, rect.y1)]
                    for line in lines[1:]:
                        rect = line['rect']
                        prev_line = top_lines[-1]
                        # If one line ends where the other starts, they are the same line
                        if np.isclose(rect.x0, prev_line[2], atol=1):
                            top_lines[-1] = (prev_line[0], prev_line[1], rect.x1, prev_line[3])
                        # If one line starts where the other ends, they are the same line
                        elif np.isclose(rect.x1, prev_line[0], atol=1):
                            top_lines[-1] = (rect.x0, prev_line[1], prev_line[2], prev_line[3])
                        # Otherwise, it's a new line
                        else:
                            top_lines.append((rect.x0, rect.y0, rect.x1, rect.y1))
                    assert len(top_lines) in [1, 2], \
                        f'Expected at most two horizontal lines for table(s) in row {row}, ' \
                        f'col {col} in page {page.number} in {self.lap_times_file}. Found ' \
                        f'{len(top_lines)}'

                    # Find the column separators
                    """
                    The left and right boundary of each table is simply the left and right end of
                    the top line. The right of column 0, which is "NO", is the right boundary of
                    the text "NO". We don't really know the right boundary for the pit column, but
                    that's roughly at the mid point of top line. Then from the mid point to the
                    right end is the "TIME" column. In practice, we use mid point - 5 as the right
                    boundary for the pit column.
                    
                    Below, `col_seps` is a list of column separators for each table. That is,
                    `col_seps[1]` gives the column separators for the second table in this row.
                    """
                    col_seps = []
                    for line in top_lines:
                        no = page.search_for('NO',clip=(line[0], line[1] - 15, line[2], line[3]))
                        assert len(no) == 1, f'Expected exactly one "NO" above the top line at ' \
                            f'({line[0], line[1], line[2], line[3]}) on p.{page.number} in ' \
                            f'{self.lap_times_file}. Found {len(no)}'
                        col_seps.append([
                            (line[0], no[0].x1),
                            (no[0].x1, (line[0] + line[2]) / 2 - 5),
                            ((line[0] + line[2]) / 2 - 5, line[2])
                        ])

                    # Find the white and grey rectangles under the top lines. Each row is either
                    # coloured/filled in white or grey, so we can get the row's top and bottom
                    # y-positions from these rectangles
                    rects = [i for i in page.get_drawings_in_bbox(bbox)
                             if i['rect'].y1 - i['rect'].y0 > 10]
                    ys = [j for i in rects for j in [i['rect'].y0, i['rect'].y1]]
                    ys.sort()
                    row_seps = [ys[0]]
                    for y in ys[1:]:
                        if y - row_seps[-1] > 10:
                            row_seps.append(y)
                    row_seps = [(row_seps[i], row_seps[i + 1]) for i in range(len(row_seps) - 1)]

                    # Finally we are good to parse the tables using these separators
                    temp = []
                    for cols in col_seps:
                        tab, superscript, cross_out = page.parse_table_by_grid(
                            vlines=cols, hlines=row_seps
                        )
                        assert len(superscript) == 0, \
                            f'Some superscript(s) found in table at ({cols[0][0]:.1f}, ' \
                            f'{row_seps[0][0]:.1f}, {cols[2][1]:.1f}, {row_seps[-1][1]:.1f}) in ' \
                            f'page {page.number} in {self.lap_times_file}. But we expect none'
                        for i, _, _ in cross_out:
                            tab.loc[i, 'lap_time_deleted'] = True
                        # Drop empty row
                        """
                        This is because the two side-by-side tables may not have the same amount of
                        rows. E.g., there are 11 laps, and the left table will have 6 and the right
                        table has 5 rows. The right table will have an empty row at the bottom, so
                        drop it here
                        """
                        tab = tab[tab[0] != '']
                        temp.append(tab)

                    # One driver may have multiple tables. Concatenate them
                    temp = pd.concat(temp, ignore_index=True)
                    temp['car_no'] = car_no
                    temp['driver'] = driver
                    df.append(temp)

        # Clean up
        df = pd.concat(df, ignore_index=True)
        if 'lap_time_deleted' not in df.columns:
            df['lap_time_deleted'] = False
        df.fillna({'lap_time_deleted': False}, inplace=True)
        df.rename(columns={0: 'lap_no', 1: 'pit', 2: 'lap_time'}, inplace=True)
        df.lap_no = df.lap_no.astype(int)
        df.car_no = df.car_no.astype(int)
        df.replace('', None, inplace=True)
        df.pit = (df.pit == 'P').astype(bool)
        df = self._assign_session_to_lap(self.classification_df, df)

        # Overwrite `.to_json()` and `.to_pkl()` methods
        # TODO: bad practice
        def to_json() -> list[dict]:
            lap_data = []
            # TODO: first lap's lap time is calendar time, not lap time, so drop to
            lap_times = df[df.lap_no >= 2].copy()
            lap_times.lap_time = lap_times.lap_time.apply(duration_to_millisecond)
            for q in [1, 2, 3]:
                temp = lap_times[lap_times.Q == q].copy()
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
                        year=self.year,
                        round=self.round_no,
                        session=f'Q{q}' if self.session == 'quali' else f'SQ{q}',
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

        def to_pkl(filename: str | os.PathLike) -> None:
            with open(filename, 'wb') as f:
                pickle.dump(to_json(), f)
            return

        df.to_json = to_json
        df.to_pkl = to_pkl
        return df


class PitStopParser:
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

    # TODO: refactor. Have too many parse table by grid, in utils, in quali. parser, here, and in
    #       entry list
    def _parse_table_by_grid(
            self,
            page: Page,
            vlines: list[float],
            hlines: list[float],
            tol: float = 2
    ) -> pd.DataFrame:
        """Manually parse the table cell by cell, defined by lines separating the columns and rows

        See `EntryListParser._parse_table_by_grid()` for detailed explanation.

        :param vlines: x-coords. of vertical lines separating the cols. Table left and right
                       boundaries need to be included
        :param hlines: y-coords. of horizontal lines separating the rows. Table top and bottom
                       boundaries need to be included
        :param tol: tolerance for bbox. of text. Default is 2 pixels
        """
        cells = []
        for i in range(len(hlines) - 1):
            row = []
            for j in range(len(vlines) - 1):
                text = ''
                l, t, r, b = vlines[j], hlines[i], vlines[j + 1], hlines[i + 1]
                cell = page.get_text('blocks', clip=(l, t, r, b))
                if cell:
                    assert len(cell) == 1, f'Expected exactly one cell in row {i}, col {j} in ' \
                                           f'{self.file}. Found: {cell}'
                    cell = cell[0]
                    if cell[4].strip():
                        bbox = cell[0:4]
                        if bbox[0] < vlines[j] - tol or bbox[2] > vlines[j + 1] + tol \
                                or bbox[1] < t - tol or bbox[3] > b + tol:
                            raise ValueError(f'Found text outside the cell in row {i}, col {j} in '
                                             f'{self.file}')
                        text = cell[4].strip()
                row.append(text)
            cells.append(row)
        return pd.DataFrame(cells[1:], columns=cells[0])

    def _parse(self) -> pd.DataFrame:
        doc = pymupdf.open(self.file)
        df = []
        # TODO: would be nice to add a test for page numbers: if more than one page, we should have
        #       "page x of xx" at the bottom right of each page
        for page in doc:  # Can have multiple pages, though usually only one. E.g., 2023 Dutch
            # Get the position of the table
            page = Page(page)
            driver = page.search_for('DRIVER')
            assert len(driver) == 1, f'Expected exactly one "DRIVER" in {self.file}. Found: ' \
                                     f'{driver}'
            driver = driver[0]
            t = driver.y0 - 1    # "DRIVER" gives the top of the table
            w = page.bound()[2]  # The right of the page

            # Find the vertical lines separating the cols
            no = page.search_for('NO', clip=(0, t, w, driver.y1))
            assert len(no) == 1, f'Expected exactly one "NO" in the header row in {self.file}. ' \
                                 f'Found: {no}'
            no = no[0]
            b_header = no.y1 + 1  # Bottom of the header row
            # Find the positions of the col. headers
            headers = ('NO', 'DRIVER', 'ENTRANT', 'LAP', 'TIME OF DAY', 'STOP', 'DURATION',
                       'TOTAL TIME')  # The col. names
            cols: dict[str, tuple[float, float]] = {}
            l = no.x0 - 1
            for col in headers:
                col_name = col
                col = page.search_for(col, clip=(l, t, w, b_header))
                assert len(col) == 1, f'Expected exactly one "{col}" in the header row in ' \
                                      f'{self.file}. Found: {col}'
                col = col[0]
                cols[col_name] = (col.x0, col.x1)
            # Now the vertical line seps. are the left and right point of each col. header
            vlines = [
                cols['NO'][0] - 1,                        # Left of "NO"
                (cols['NO'][1] + cols['DRIVER'][0]) / 2,  # Between "NO" and "DRIVER"
                cols['ENTRANT'][0] - 1,                   # Left of "ENTRANT"
                cols['LAP'][0] - 1,                       # Left of "LAP"
                cols['LAP'][1] + 1,                       # Right of "LAP"
                cols['TIME OF DAY'][1],                   # Right of "TIME OF DAY"
                cols['STOP'][1],                          # Right of "STOP"
                cols['DURATION'][1],                      # Left of "TOTAL TIME"
                cols['TOTAL TIME'][1]                     # Right of "TOTAL TIME"
            ]

            # Get the bottom of the table. We identify page bottom by a white blank strip
            pixmap = page.get_pixmap(clip=(0, t + 10, w, page.bound()[3]))
            l, t, r, b = pixmap.x, pixmap.y, pixmap.x + pixmap.w, pixmap.y + pixmap.h
            pixmap = np.ndarray([b - t, r - l, 3], dtype=np.uint8, buffer=pixmap.samples_mv)
            is_white_row = np.all(pixmap == 255, axis=(1, 2))
            white_strips = []
            strip_start = None
            for i, is_white in enumerate(is_white_row):
                if is_white and strip_start is None:
                    strip_start = i
                elif not is_white and strip_start is not None:
                    if i - strip_start >= 10:  # At least 10 rows of white
                        white_strips.append(strip_start + t)
                    strip_start = None
            # If the strip is at the bottom. Shouldn't happen but just in case
            if strip_start is not None and len(is_white_row) - strip_start >= 10:
                white_strips.append(strip_start + t)
            if not white_strips:
                raise ValueError(f'Could not find a blank white strip in {self.file}')
            b = white_strips[0] + 1

            # Get the row positions. The rows are coloured in grey, white, grey, white, so get the
            # top and bottom positions of the grey rectangles
            rects = []
            for i in page.get_drawings_in_bbox(bbox=(0, b_header - 1, w, b)):
                if i['fill'] is not None and np.allclose(i['fill'], 0.9, rtol=0.05):
                    rects.append(i['rect'].y0 + 1)
                    rects.append(i['rect'].y1 - 1)
            rects.sort()
            hlines: list[float] = [no.y0 - 1, b_header]
            for i in rects:
                if i - hlines[-1] > 5:
                    hlines.append(i)
            if b - hlines[-1] > 5:
                hlines.append(b)

            # Parse
            tab = self._parse_table_by_grid(page, vlines, hlines)
            df.append(tab)

        # Clean up the table
        df = pd.concat(df, ignore_index=True)
        df.dropna(subset=['NO'], inplace=True)  # Drop empty rows, if any
        df = df[df.NO != '']
        df = df[['NO', 'LAP', 'TIME OF DAY', 'STOP', 'DURATION']].reset_index(drop=True)
        df.rename(columns={
            'NO': 'car_no',
            'LAP': 'lap',
            'TIME OF DAY': 'local_time',
            'STOP': 'stop_no',
            'DURATION': 'duration'
        }, inplace=True)
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

        def to_pkl(filename: str | os.PathLike) -> None:
            with open(filename, 'wb') as f:
                pickle.dump(df.to_json(), f)
            return

        df.to_json = to_json
        df.to_pkl = to_pkl
        return df


if __name__ == '__main__':
    pass
