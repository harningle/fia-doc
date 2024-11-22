# -*- coding: utf-8 -*-
import os
import pickle
import re
from typing import Literal
import warnings

import numpy as np
import pandas as pd
import pymupdf

from .models.classification import(
    Classification,
    ClassificationData,
    QualiClassification,
    QualiClassificationData
)
from .models.driver import Driver, DriverData
from .models.foreign_key import RoundEntry, SessionEntry
from .models.lap import Lap, LapData, QualiLap
from .models.pit_stop import PitStop, PitStopData
from .utils import duration_to_millisecond, time_to_timedelta

pd.set_option('future.no_silent_downcasting', True)


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

    def _parse_table_by_grid(self, page: pymupdf.Page, vlines: list[float], hlines: list[float]) \
            -> pd.DataFrame:
        """Manually parse the table cell by cell, defined by lines separating the columns and rows

        The reason why we parse this table manually rather than using `page.find_tables` is that
        when we have the reserve driver table, car No. may have superscript, which can not be
        handled otherwise. The superscript indicates which reserve driver is driving whose case.
        E.g., Antonelli is driving Hamilton's car, then driver No. 44 and driver No. 12 will have
        the same superscript.

        :param vlines: x-coords. of vertical lines separating the cols.
        :param hlines: y-coords. of horizontal lines separating the rows
        :return:
        """
        cells = []
        vgap = vlines[1] - vlines[0]  # Usual gap between two vertical lines
        for i in range(len(hlines) - 1):
            row = []
            has_superscript = False
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
                            b = hlines[i] + vgap + 2
                        else:  # The unusual big gap is above, so now at the reserve driver table
                            t = hlines[i + 1] - vgap - 2
                            b = hlines[i + 1]

                # Get text in the cell
                # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html
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
                                # PyMuPDF is notoriously bad for respecting `clip` parameter. We
                                # give one pixel tolerance
                                if bbox[0] >= vlines[j] - 2 \
                                        and bbox[2] <= vlines[j + 1] + 2 \
                                        and bbox[1] >= t - 2 \
                                        and bbox[3] <= b + 2:
                                    spans.append(span)

                # Check if any superscript
                # See https://pymupdf.readthedocs.io/en/latest/recipes-text.html#how-to-analyze-
                # font-characteristics for font flags
                match len(spans):
                    case 1:
                        row.append(spans[0]['text'].strip())
                    case 2:
                        for span in spans:
                            match span['flags']:
                                case 0:
                                    row.append(span['text'].strip())
                                case 1:
                                    has_superscript = True
                                    superscript = span['text'].strip()
                                case _:
                                    raise ValueError(f'Unknown error when parsing row {i}, col '
                                                     f'{j} in {self.file}')
                    case _:
                        raise ValueError(f'Unknown error when parsing row {i}, col {j} in '
                                         f'{self.file}')
            if has_superscript:
                row.append(superscript)
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
            assert len(temp) == 2, f'Expected 2 rows for superscript {i}, got {len(temp)}'
            assert temp.car_no.nunique() == 2, \
                f'Expected 2 different drivers for superscript {i}, got {temp.car_no.nunique()}'
            df.loc[df[df.reserve == i].index[1], 'reserve_for'] = temp['driver'].iloc[0]
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
        df = self._parse_table_by_grid(page, aux_vlines, aux_hlines)

        def to_json() -> list[dict]:
            return [
                DriverData(
                    foreign_keys=RoundEntry(
                        year=self.year,
                        round=self.round_no,
                        team_name=x.constructor,
                        driver_name=x.driver
                    ),
                    objects=[
                        Driver(
                            car_number=x.car_no
                        )
                    ]
                ).model_dump()
                for x in df.itertuples()
            ]

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
            history_chart_file: str | os.PathLike,
            year: int,
            round_no: int,
            session: Literal['race', 'sprint_race']
    ):
        self.classification_file = classification_file
        self.history_chart_file = history_chart_file
        self.session = session
        self.year = year
        self.round_no = round_no
        self._check_session()
        self.classification_df = self._parse_classification()
        self.lap_times_df = self._parse_lap_times()
        # self._cross_validate()

    def _check_session(self) -> None:
        """Check that the input session is valid. Raise an error otherwise"""
        if self.session not in ['race', 'sprint_race']:
            raise ValueError(f'Invalid session: {self.session}. Valid sessions are: "race" and '
                             f'"sprint_race""')
        return

    def _parse_classification(self) -> pd.DataFrame:
        """Parse "Race/Sprint Race Final Classification" PDF

        The output dataframe has columns [driver No., laps completed, total time,
        finishing position, finishing status, fastest lap time, fastest lap speed, fastest lap No.]
        """
        # Find the page with "Final Classification", on which the table is located
        doc = pymupdf.open(self.classification_file)
        found = []
        for i in range(len(doc)):
            page = doc[i]
            found = page.search_for('Final Classification')
            if found:
                break
            found = page.search_for('Provisional Classification')
            if found:
                warnings.warn('Found and using provisional classification, not the final one')
                break
        if not found:
            doc.close()
            raise ValueError(f'"Final Classification" not found on any page in '
                             f'{self.classification_file}')

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

        # Auxiliary vertical lines separating the columns
        aux_lines = [
            pos['NO']['left'],
            (pos['NO']['right'] + pos['DRIVER']['left']) / 2,
            pos['NAT']['left'],
            pos['NAT']['right'],
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

        # Find the table using the bounding box and auxiliary lines above
        """
        TODO: We fine tuned `snap_x_tolerance` a bit. Otherwise, pymupdf would think there is
        another (empty) col. between "FASTEST" and "ON". Can be a bit fragile.
        """
        df = page.find_tables(
            clip=bbox,
            strategy='lines',
            vertical_lines=aux_lines,
            snap_x_tolerance=pos['ON']['left'] - pos['FASTEST']['right']
        )
        assert len(df.tables) == 1, \
            f'Expected one table, got {len(df.tables)} in {self.classification_file}'
        df = df[0].to_pandas()
        df = df[(df.NO != '') | df.NO.isnull()]  # May get some empty rows at the bottom. Drop them
        assert df.shape[1] == 13, \
            f'Expected 13 columns, got {df.shape[1]} in {self.classification_file}'

        # Do the same for the "NOT CLASSIFIED" table
        if has_not_classified:
            t = page.search_for('NOT CLASSIFIED')[0].y1
            b = page.search_for('FASTEST LAP')[0].y0
            not_classified = page.find_tables(
                clip=(0, t, w, b),
                strategy='lines',
                vertical_lines=aux_lines,
                snap_x_tolerance=pos['ON']['left'] - pos['FASTEST']['right']
            )
            assert len(not_classified.tables) == 1, \
                f'Expected one table for "NOT CLASSIFIED", got {len(not_classified.tables)} ' \
                f'in {self.classification_file}'
            not_classified = not_classified[0].to_pandas()

            # The table header is actually the first row of the "NOT CLASSIFIED" table
            not_classified.loc[-1] = not_classified.columns
            not_classified.sort_index(inplace=True)
            not_classified.reset_index(drop=True, inplace=True)
            assert not_classified.shape[1] == 13, \
                f'Expected 13 columns for "NOT CLASSIFIED" table , got ' \
                f'{not_classified.shape[1]} in {self.classification_file}'
            not_classified.columns = df.columns
            not_classified = not_classified[(not_classified.NO != '') | not_classified.NO.isnull()]
            not_classified['finishing_status'] = 11  # TODO: should clean up the code later
            not_classified['is_classified'] = False
            df = pd.concat([df, not_classified], ignore_index=True)

        # Set col. names
        del df['NAT']
        df.rename(columns={
            'Col0':    'finishing_position',
            'NO':      'car_no',
            'DRIVER':  'driver',
            'ENTRANT': 'team',
            'LAPS':    'laps_completed',
            'TIME':    'time',            # How long it took the driver to finish the race
            'GAP':     'gap',
            'INT':     'int',
            'KM/H':    'avg_speed',
            'FASTEST': 'fastest_lap_time',
            'ON':      'fastest_lap_no',  # The lap number on which the fastest lap was set
            'PTS':     'points'
        }, inplace=True)
        df.replace({'': None}, inplace=True)  # Empty string --> `None`, so `pd.isnull` works

        # Remove the "Colx" in cells. These are col. name placeholders from the parsing above
        df.replace(r'Col\d+', None, regex=True, inplace=True)

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
        df.sort_values(by='temp', inplace=True)
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
            'is_classified': True,
            'points': 0,
            'finishing_status': 0
        }, inplace=True)
        df.finishing_status = df.finishing_status.astype(int)

        def to_json() -> list[dict]:
            return df.apply(
                lambda x: ClassificationData(
                    foreign_keys=SessionEntry(
                        year=self.year,
                        round=self.round_no,
                        session=self.session,
                        car_number=x.car_no
                    ),
                    objects=[
                        Classification(
                            position=x.finishing_position,
                            is_classified=x.is_classified,
                            status=x.finishing_status,
                            points=x.points,
                            time=x.time,
                            laps_completed=x.laps_completed,
                            fastest_lap_rank=x.fastest_lap_rank if x.fastest_lap_time else None
                            # TODO: replace the rank with missing or -1 in self.classification_df
                        )
                    ]
                ).model_dump(exclude_none=True),
                axis=1
            ).tolist()

        def to_pkl(filename: str | os.PathLike) -> None:
            with open(filename, 'wb') as f:
                pickle.dump(df.to_json(), f)
            return

        df.to_json = to_json
        df.to_pkl = to_pkl
        return df

    def _parse_lap_times(self) -> pd.DataFrame:
        doc = pymupdf.open(self.history_chart_file)
        df = []
        for page in doc:
            # Each page can have multiple tables, all of which begins from the same top y-position.
            # Their table headers are vertically bounded between "History Chart" and "TIME". Find
            # all of the headers
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

        # Merge in the fastest lap info. from final classification
        df = df.merge(self.classification_df[['car_no', 'fastest_lap_time', 'fastest_lap_no']],
                      on='car_no',
                      how='left')
        temp = df[df.lap == df.fastest_lap_no]
        assert (temp.time == temp.fastest_lap_time).all(), \
            'Fastest lap time in lap times does not match the one in final classification'
        df['is_fastest_lap'] = df.lap == df.fastest_lap_no
        del df['fastest_lap_time']

        def to_json() -> list[dict]:
            temp = df.copy()
            temp.lap = temp.apply(
                lambda x: Lap(
                    number=x.lap,
                    position=x.position,
                    time=duration_to_millisecond(x.time),
                    is_entry_fastest_lap=x.is_fastest_lap
                ),
                axis=1
            )
            temp = temp.groupby('car_no')[['lap']].agg(list).reset_index()
            temp['session_entry'] = temp.car_no.map(
                lambda x: SessionEntry(
                    year=self.year,
                    round=self.round_no,
                    session='R' if self.session == 'race' else 'SR',
                    car_number=x
                )
            )
            return temp.apply(
                lambda x: LapData(
                    foreign_keys=x.session_entry,
                    objects=x.lap
                ).model_dump(),
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
            session: Literal['quali', 'sprint_quali']
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
        if self.session not in ['quali', 'sprint_quali']:
            raise ValueError(f'Invalid session: {self.session}. Valid sessions are: "quali" and '
                             f'"sprint_quali""')
        # TODO: 2023 US sprint shootout. No "POLE POSITION LAP"???
        return

    def _parse_classification(self):
        # Find the page with "Qualifying Session Final Classification"
        doc = pymupdf.open(self.classification_file)
        found = []
        for i in range(len(doc)):
            page = doc[i]
            found = page.search_for('Final Classification')
            if found:
                break
            found = page.search_for('Provisional Classification')
            if found:
                warnings.warn('Found and using provisional classification, not the final one')
                break
        if not found:
            raise ValueError(f'"Final Classification" not found on any page in '
                             f'{self.classification_file}')

        # Page width. This is the rightmost x-coord. of the table
        w = page.bound()[2]

        # y-position of "Final Classification", which is the topmost y-coord. of the table
        y = found[0].y1

        # y-position of "NOT CLASSIFIED - " or "POLE POSITION LAP", whichever comes the first. This
        # is the bottom of the classification table
        has_not_classified = False
        bottom = page.search_for('NOT CLASSIFIED - ')
        if bottom:
            has_not_classified = True
        else:
            bottom = page.search_for('POLE POSITION LAP')
        if not bottom:
            raise ValueError(f'Could not find "NOT CLASSIFIED - " or "POLE POSITION LAP" in '
                             f'{self.classification_file}')
        b = bottom[0].y0

        # Table bounding box
        bbox = pymupdf.Rect(0, y, w, b)

        # Dist. between "NAT" and "ENTRANT"
        nat = page.search_for('NAT')[0]
        entrant = page.search_for('ENTRANT')[0]
        snap_x_tolerance = (entrant.x0 - nat.x1) * 1.2  # 20% buffer. TODO: fragile

        # Get the table
        df = page.find_tables(clip=bbox, snap_x_tolerance=snap_x_tolerance)
        assert len(df.tables) == 1, \
            f'Expected one table, got {len(df.tables)} in {self.classification_file}'
        aux_lines = sorted(set([round(i[0], 2) for i in df[0].cells]))  # For unclassified table
        df = df[0].to_pandas()
        # TODO: check 2023 vs 2024 PDF. Do we have a "%" col.? 15 or 14 col. in total?
        assert df.shape[1] == 14, \
            f'Expected 15 columns, got {df.shape[1]} in {self.classification_file}'

        # Clean up column name: the first row is mistakenly taken as column names
        """
        TODO: need to check if the first row is correctly treated as the table content, or
        mistakenly treated as col. header. Can be checked by the top y-position of `tableheader`
        from `page.find_tables()`. If the y-position exceeds the `y`, then it's a mistake.`
        
        Also, we name the sessions as "Q1", "Q2", and "Q3", regardless of whether it's a normal
        qualifying or a sprint qualifying. This makes the code simpler, and we should always use
        `self.session` to determine what session it is.
        """
        cols = df.columns.tolist()
        for i in range(len(df.columns)):
            cols[i] = df.columns[i].removeprefix(f'{i}-')
        df = pd.DataFrame(
            np.vstack([cols, df]),
            columns=['position', 'NO', 'DRIVER', 'NAT', 'ENTRANT', 'Q1', 'Q1_LAPS', 'Q1_TIME',
                     'Q2', 'Q2_LAPS', 'Q2_TIME', 'Q3', 'Q3_LAPS', 'Q3_TIME']
        )
        df = df[(df.NO != '') | df.NO.isnull()]  # May get some empty rows at the bottom. Drop them
        df['finishing_status'] = 0

        # Do the same for the "NOT CLASSIFIED" table
        # TODO: this can fail when the "NOT CLASSIFIED" table is very short, e.g. only one row
        if has_not_classified:
            t = page.search_for('NOT CLASSIFIED - ')[0].y1
            b = page.search_for('POLE POSITION LAP')[0].y0
            not_classified = page.find_tables(
                clip=pymupdf.Rect(0, t, w, b),
                strategy='lines',
                vertical_lines=aux_lines,
                add_lines=[((0, t), (w, t)), ((0, b), (w, b))]
            )
            assert len(not_classified.tables) == 1, \
                f'Expected one table for "NOT CLASSIFIED", got {len(not_classified.tables)} ' \
                f'in {self.file}'
            not_classified = not_classified[0].to_pandas()
            not_classified.loc[-1] = not_classified.columns.str.replace(r'Col\d+', '', regex=True)
            not_classified.sort_index(inplace=True)
            not_classified.reset_index(drop=True, inplace=True)
            assert not_classified.shape[1] == 15, \
                f'Expected 15 columns for "NOT CLASSIFIED"table , got {not_classified.shape[1]} ' \
                f'in {self.file}'
            not_classified['finishing_status'] = 11  # TODO: should clean up the code later
            not_classified.columns = df.columns
            not_classified = not_classified[(not_classified.NO != '') | not_classified.NO.isnull()]
            df = pd.concat([df, not_classified], ignore_index=True)

            # Fill in the position for DNF and DSQ drivers
            # TODO: should find a PDF with DSQ drivers to handle such cases
            df.replace({'': None, 'DNF': None}, inplace=True)
            df.position = df.position.astype(float).ffill() + df.position.isnull().cumsum()
            df.position = df.position.astype(int)

        # Clean up
        df.NO = df.NO.astype(int)
        del df['NAT']
        df.replace('', None, inplace=True)
        df.position = df.position.astype(int)

        # Overwrite `.to_json()` and `.to_pkl()` methods
        # TODO: bad practice
        def to_json() -> list[dict]:
            data = []
            for q in [1, 2, 3]:
                temp = df[df[f'Q{q}_TIME'].notnull()][['position', 'NO', f'Q{q}_TIME']].copy()
                temp['classification'] = temp.apply(
                    lambda x: QualiClassificationData(
                        foreign_keys=SessionEntry(
                            year=self.year,
                            round=self.round_no,
                            session=f'Q{q}' if self.session == 'quali' else f'SQ{q}',
                            car_number=x.NO
                        ),
                        objects=[
                            QualiClassification(
                                position=x.position
                            )
                        ]
                    ).model_dump(),
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
    def get_strikeout_text(page: pymupdf.Page) -> list[tuple[pymupdf.Rect, str]]:
        """Get all strikeout texts and their locations in the page

        See https://stackoverflow.com/a/74582342/12867291.

        :param page: fitz.Page object
        :return: A list of tuples, where each tuple is the bbox and text of a strikeout text
        """
        # Get all strikeout lines
        lines = []
        paths = page.get_drawings()  # Strikeout lines are in fact vector graphics. To be more
        for path in paths:           # precise, they are short rectangles with very small height
            for item in path['items']:
                if item[0] == 're':  # If a graphic is a rect., check its height: absolute height
                    rect = item[1]   # should < 1px, and have some sizable width relative to height
                    if (rect.width > 2 * rect.height) and (rect.height < 1):
                        lines.append(rect)

        # Get all texts on this page
        # TODO: the O(n^2) here can probably be optimised later
        words = page.get_text('words')
        strikeout = []
        for rect in lines:
            for w in words:  # `w` is a iterable `(x0, y0, x1, y1, text)`
                text_rect = pymupdf.Rect(w[:4])     # Location/bbox of the word
                if text_rect.intersects(rect):      # If the word's location intersects with a
                    strikeout.append((rect, w[4]))  # strikeout line, it's a strikeout text
        return strikeout

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
        """Parse "Qualifying/Sprint Quali./Shootout Session Lap Times" PDF

        This is the most complicated parsing. See `notebook/demo.ipynb` for more details.
        """
        doc = pymupdf.open(self.lap_times_file)
        df = []
        for page in doc:
            # Page width. All tables are horizontally bounded in [0, `w`]
            w = page.bound()[2]

            # Positions of "NO" and "TIME". They are the top of each table. One driver may have
            # multiple tables starting from roughly the same top y-position
            no_time_pos = page.search_for('NO TIME')
            assert len(no_time_pos) >= 1, \
                f'Expected at least one "NO TIME", got {len(no_time_pos)} in {self.lap_times_file}'
            ys = [i.y1 for i in no_time_pos]
            ys.sort()                      # Sort these "NO"'s from top to bottom
            top_pos = [ys[0] - 1]          # -1 to give a little buffer at the top
            for y in ys[1:]:
                # Many "NO"'s are roughly at the same height (usually three drivers share the full
                # width of the page, and each of them have two tables side by side, so six tables
                # and six "NO"'s are vertically at the same y-position). We only need those at
                # different y-positions. If there is a 10+ px vertical gap, we take it as a "NO" at
                # a lower y-position
                if y - top_pos[-1] > 10:
                    top_pos.append(y - 1)  # Again, -1 to allow a little buffer at the top

            # Bottom of the table is the next "NO TIME", or the bottom of the page
            ys = [i.y0 for i in no_time_pos]
            ys.sort()
            bottom_pos = [ys[0]]
            for y in ys[1:]:
                if y - bottom_pos[-1] > 10:
                    bottom_pos.append(y)
            b = page.bound()[3]
            bottom_pos.append(b)
            bottom_pos = bottom_pos[1:]    # The first "NO TIME" is not the bottom of any table

            # Find the tables located between each `top_pos` and `bottom_pos`
            for row in range(len(top_pos)):
                for col in range(3):            # Three drivers in one row
                    # Find the driver name, which is located immediately above the table
                    driver = page.get_text(
                        'block',
                        clip=(
                            col * w / 3,        # Each driver occupies 1/3 of the page width
                            top_pos[row] - 30,  # Driver name is usually 20-30 px above the table
                            (col + 1) * w / 3,
                            top_pos[row] - 10
                        )
                    ).strip()
                    if not driver:  # In the very last row may not have all three drivers. E.g., 20
                        continue    # drivers, 3 per row, so the last row only has 2 drivers
                                    # TODO: may want a test here. Every row above should have
                                    # precisely 3 drivers
                    car_no, driver = driver.split(maxsplit=1)

                    # Find tables in the bounding box of driver i's table
                    tabs = page.find_tables(
                        clip=(
                            col * w / 3,
                            top_pos[row],
                            (col + 1) * w / 3,
                            bottom_pos[row]
                        ),
                        strategy='lines'
                    )  # TODO: I think this may fail if a driver only has one lap and crashes...

                    # Check if any lap times is strikeout
                    # TODO: the O(n^2) here should be optimised later. We are processing the same
                    #       page multiple times
                    strikeout_texts = self.get_strikeout_text(page)
                    temp = []
                    lap_time_deleted = []
                    for tab in tabs:
                        for r in tab.rows:
                            assert len(r.cells) == 3  # Should have lap, if pit, and lap time col.
                            cell = r.cells[2]         # Only lap time cell (the last cell) can be
                            is_strikeout = False      # strikeout and needs to be checked
                            bbox = pymupdf.Rect(cell)
                            for rect, text in strikeout_texts:
                                if bbox.intersects(rect):
                                    is_strikeout = True
                                    break
                            lap_time_deleted.append(is_strikeout)
                        temp.append(tab.to_pandas())

                    # One driver may have multiple tables. Concatenate them
                    temp = pd.concat(temp, ignore_index=True)
                    temp.columns = ['lap_no', 'pit', 'lap_time']
                    temp['car_no'] = car_no
                    temp['driver'] = driver
                    temp['lap_time_deleted'] = lap_time_deleted
                    df.append(temp)

        # Clean up
        df = pd.concat(df, ignore_index=True)
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
                    lambda x: QualiLap(
                        number=x.lap_no,
                        time=x.lap_time,
                        is_deleted=x.lap_time_deleted,
                        is_entry_fastest_lap=x.is_fastest_lap
                    ),
                    axis=1
                )
                temp = temp.groupby('car_no')[['lap']].agg(list).reset_index()
                temp['session_entry'] = temp['car_no'].map(
                    lambda x: SessionEntry(
                        year=self.year,
                        round=self.round_no,
                        session=f'Q{q}' if self.session == 'quali' else f'SQ{q}',
                        car_number=x
                    )
                )
                temp['lap_data'] = temp.apply(
                    lambda x: LapData(
                        foreign_keys=x['session_entry'],
                        objects=x['lap']
                    ).model_dump(),
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
            session: Literal['race', 'sprint']
    ):
        self.file = file
        self.year = year
        self.round_no = round_no
        self.session = session
        self._check_session()
        self.df = self._parse()

    def _check_session(self) -> None:
        if self.session not in ['race', 'sprint']:
            raise ValueError(f'Invalid session: {self.session}. Valid sessions are '
                             f'"race" and "sprint"')
        return

    def _parse(self) -> pd.DataFrame:
        doc = pymupdf.open(self.file)
        page = doc[0]
        # TODO: have PDFs containing multiple pages?

        # Get the position of the table
        t = page.search_for('DRIVER')[0].y0      # "DRIVER" gives the top of the table
        w, h = page.bound()[2], page.bound()[3]  # Page right and bottom boundaries are the table's
                                                 # as well
        # Parse
        df = page.find_tables(clip=(0, t, w, h), strategy='lines')
        assert len(df.tables) == 1, f'Expected one table, got {len(df.tables)} in {self.file}'
        df = df[0].to_pandas()

        # Clean up the table
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
                lambda x: PitStop(
                    lap=x.lap,
                    number=x.stop_no,
                    duration=duration_to_millisecond(x.duration),
                    local_timestamp=x.local_time
                ),
                axis=1
            )
            pit_stop = pit_stop.groupby('car_no')[['pit_stop']].agg(list).reset_index()
            pit_stop['session_entry'] = pit_stop.car_no.map(
                lambda x: SessionEntry(
                    year=self.year,
                    round=self.round_no,
                    session=self.session if self.session == 'race' else 'SR',
                    car_number=x
                )
            )
            return pit_stop.apply(
                lambda x: PitStopData(
                    foreign_keys=x.session_entry,
                    objects=x.pit_stop
                ).model_dump(),
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
