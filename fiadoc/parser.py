# -*- coding: utf-8 -*-
import os
import pickle
from typing import Literal

import numpy as np
import pandas as pd
import pymupdf

from .models.foreign_key import SessionEntry
from .models.classification import Classification, ClassificationData
from .utils import duration_to_millisecond


class ClassificationParser:
    def __init__(
            self,
            file: str | os.PathLike,
            year: int,
            round_no: int,
            session: Literal['practice', 'quali', 'race', 'sprint_quali', 'sprint_race']
    ):
        self.file = file
        self.session = session
        self.year = year
        self.round_no = round_no
        self._check_session()
        self.df = self._parse()

    def _check_session(self) -> None:
        """Check that the input session is valid. Raise an error otherwise"""
        if self.session not in ['quali', 'race', 'sprint_quali', 'sprint_race', 'practice']:
            raise ValueError(f'Invalid session: {self.session}. Valid sessions are: "quali", '
                             f'"race", "sprint_quali", "sprint_race", and "practice"')
        if self.session == 'practice':
            raise NotImplementedError('Practice sessions are not supported yet')
        return

    def _parse(self) -> pd.DataFrame:
        """Parse "Race/Qualifying/Sprint Race/Sprint Qualifying Final Classification" PDF

        The output dataframe has columns [driver No., laps completed, total time,
        finishing position, finishing status, fastest lap time, fastest lap speed, fastest lap No.]
        """
        # Find the page with "Final Classification", on which the table is located
        doc = pymupdf.open(self.file)
        found = []
        for i in range(len(doc)):
            page = doc[i]
            found = page.search_for('Final Classification')
            if found:
                break
        if not found:
            doc.close()
            raise ValueError(f'"Final Classification" not found on any page in {self.file}')

        # Page width. This is the rightmost x-coord. of the table
        w = page.bound()[2]

        # Position of "Final Classification". Topmost y-coord. of the table
        y = found[0].y1

        # Bottommost y-coord. of the table
        """
        Depending on which session is being parsed, the bottom of the table is identified by:

        * race/sprint race: "NOT CLASSIFIED" or "FASTEST LAP", whichever comes first
        * quali/sprint quali.: "NOT CLASSIFIED" or "POLE POSITION LAP", whichever comes first
        """
        bottom = []
        has_not_classified = False
        if self.session in ['race', 'sprint_race']:
            bottom = page.search_for('NOT CLASSIFIED')
            if not bottom:
                bottom = page.search_for('OVERALL FASTEST LAP')
            else:
                has_not_classified = True
        elif self.session in ['quali', 'sprint_quali']:
            bottom = page.search_for('NOT CLASSIFIED')
            if not bottom:
                bottom = page.search_for('POLE POSITION LAP')
            else:
                has_not_classified = True
        else:
            raise ValueError(f'Invalid session: {self.session}. Valid sessions are: "quali", '
                             f'"race", "sprint_quali", "sprint_race", and "practice"')
        if not bottom:
            raise ValueError(f'Could not find the bottom of the table in {self.file}')
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
        # TODO: sprint/quali. need different aux lines

        # Find the table using the bounding box and auxiliary lines above
        """
        TODO: We fine tuned `snap_x_tolerance` a bit. Otherwise, pymupdf would think there is
        another col. between "FASTEST" and "ON". Can be a bit fragile.
        """
        df = page.find_tables(
            clip=bbox,
            strategy='lines',
            vertical_lines=aux_lines,
            snap_x_tolerance=pos['ON']['left'] - pos['FASTEST']['right']
        )
        assert len(df.tables) == 1, f'Expected one table, got {len(df.tables)} in {self.file}'
        df = df[0].to_pandas()
        df = df[(df.NO != '') | df.NO.isnull()]  # May get some empty rows at the bottom. Drop them
        assert df.shape[1] == 13, f'Expected 13 columns, got {df.shape[1]} in {self.file}'

        # Do the same for the "NOT CLASSIFIED" table
        if has_not_classified:
            t = page.search_for('NOT CLASSIFIED')[0].y1
            b = page.search_for('OVERALL FASTEST LAP')[0].y0
            not_classified = page.find_tables(
                clip=pymupdf.Rect(0, t, w, b),
                strategy='lines',
                vertical_lines=aux_lines,
                snap_x_tolerance=pos['ON']['left'] - pos['FASTEST']['right']
            )
            assert len(not_classified.tables) == 1, \
                f'Expected one table for "NOT CLASSIFIED", got {len(not_classified.tables)} ' \
                f'in {self.file}'
            not_classified = not_classified[0].to_pandas()

            # The table header is actually the first row of the "NOT CLASSIFIED" table
            not_classified.loc[-1] = not_classified.columns
            not_classified.sort_index(inplace=True)
            not_classified.reset_index(drop=True, inplace=True)
            assert not_classified.shape[1] == 13, \
                f'Expected 13 columns for "NOT CLASSIFIED", got {not_classified.shape[1]} in ' \
                f'{self.file}'
            not_classified.columns = df.columns
            not_classified = not_classified[(not_classified.NO != '') | not_classified.NO.isnull()]
            not_classified['finishing_status'] = 11  # TODO: will clean up the code later
            df = pd.concat([df, not_classified], ignore_index=True)
        return self._clean_df(df)

    def _clean_df(self, df: pd.DataFrame) -> pd.DataFrame:
        # Rename cols.
        del df['NAT']
        df.rename(columns={
            'Col0':    'finishing_position',
            'NO':      'driver_no',
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
        df['is_classified'] = (df.finishing_position != 'DQ')
        df.loc[df.finishing_position == 'DQ', 'finishing_status'] = 20  # TODO: clean up the coding

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

        df.driver_no = df.driver_no.astype(int)
        df.laps_completed = df.laps_completed.astype(int)
        df.time = df.time.apply(duration_to_millisecond)
        # TODO: gap to the leader is to be cleaned later, so we can use it for cross validation

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
        df.fastest_lap_no = df.fastest_lap_no.astype(int)
        df['fastest_lap_rank'] = df \
            .sort_values(by=['fastest_lap_time', 'fastest_lap_no'], ascending=[True, True]) \
            .groupby('driver_no', sort=False) \
            .ngroup() + 1

        # Fill in some default values
        df.fillna({
            'is_classified': True,
            'points': 0,
            'finishing_status': 0
        }, inplace=True)
        df.finishing_status = df.finishing_status.astype(int)
        return df

    def _cross_validate(self) -> bool:
        """Cross validate against other PDFs or fastf1?"""
        raise NotImplementedError

    def to_pkl(self, filename: str | os.PathLike) -> None:
        df = self.df.copy()  # TODO: not the best practice?
        df['classification'] = df.apply(
            lambda x: ClassificationData(
                foreign_keys=SessionEntry(
                    year=self.year,
                    round=self.round_no,
                    session=self.session,
                    car_number=x.driver_no
                ),
                objects=[
                    Classification(
                        position=x.finishing_position,
                        is_classified=x.is_classified,
                        status=x.finishing_status,
                        points=x.points,
                        time=x.time,
                        fastest_lap=x.fastest_lap_no,
                        fastest_lap_rank=x.fastest_lap_rank,
                        laps_completed=x.laps_completed
                    )
                ]
            ).model_dump(),
            axis=1
        )

        # Dump to json
        classification_data = df['classification'].tolist()
        with open(filename, 'wb') as f:
            pickle.dump(classification_data, f)
        return


if __name__ == '__main__':
    pass
