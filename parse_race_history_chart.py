# -*- coding: utf-8 -*-
import datetime
import os
import pickle
import re
import warnings

import fitz
import pandas as pd

from models.lap import Lap, LapData, SessionEntry


def parse_race_history_chart_page(page: fitz.Page, page_width: float, page_height: float) -> pd.DataFrame:
    """
    Get the table(s) from a given page in "Race History Chart" PDF. There are multiple tables in a
    page, each of which correspond to a lap No. We concat all tables into one single dataframe

    See `notebook/demo.ipynb` for the detailed explanation of the table structure.

    :param page: A `fitz.Page` object
    :return: A dataframe of [driver No., lap No., gap to leader, lap time]

    TODO: probably use better type hint using pandera later
    """
    # Get the position of "Lap x"
    t = page.search_for('Race History Chart')[0].y1
    b = page.search_for('TIME')[0].y1
    headers = page.search_for('Lap', clip=(0, t, page_width, b))

    # Iterate through the tables for each lap
    tables = []
    for i, lap in enumerate(headers):
        """
        The left boundary of the table is the leftmost of the "Lap x" text, and the right boundary
        is the leftmost of the next "Lap x" text. If it's the last lap, i.e. no next table, then
        the right boundary can be determined by left boundary plus table width, which is roughly
        one-fifth of the page width. We add 5% extra buffer to the right boundary
        """
        left_boundary  = lap.x0
        right_boundary  = headers[i + 1].x0 if i + 1 < len(headers) else (left_boundary + page_width / 5) * 1.05
        temp = page.find_tables(clip=fitz.Rect(left_boundary, t, right_boundary, page_height),
                                strategy='lines',
                                add_lines=[((left_boundary, 0), (left_boundary, page_height))])[0].to_pandas()

        # Three columns: "LAP x", "GAP", "TIME". "LAP x" is the column for driver No. So add a new
        # column for lap No. with value "x", and rename the columns
        lap_no = int(temp.columns[0].split(' ')[1])
        temp.columns = ['driver_no', 'gap', 'time']
        temp['lap'] = lap_no
        temp = temp[temp['driver_no'] != '']  # Sometimes we will get one additional empty row
        temp['driver_no'] = temp['driver_no'].apply(lambda x: int(x))

        # The row order/index is meaningful: it's the order/positions of the cars
        # TODO: is this true for all cases? E.g. retirements?
        temp.reset_index(drop=False, names=['position'], inplace=True)
        temp['position'] += 1  # 1-indexed
        tables.append(temp)
    return pd.concat(tables, ignore_index=True)


def parse_race_history_chart(file: str | os.PathLike[str]) -> pd.DataFrame:
    """
    Parse "Race History Chart" PDF

    :param file: Path to PDF file
    :return: The output dataframe will be [driver No., lap No., gap to leader, lap time]
    """

    # Get page width and height
    doc = fitz.open(file)
    page = doc[0]
    page_width = page.bound()[2]
    page_height = page.bound()[3]

    # Parse all pages
    df = pd.concat([parse_race_history_chart_page(page, page_width, page_height) for page in doc], ignore_index=True)

    # Clean up
    """
    There is one tricky place: when a car is lapped (GAP col. is "LAP"), he actual lap number for
    the lapped car should be the lap number in PDF minus the #. of laps being lapped. I.e., when
    the leader starts lap 10, the lapped car starts lap 9.
    
    The lapping itself is easy to fix, but when a lapped car is in pit stop, the PDF shows "PIT" in
    the GAP col., so we cannot distinguish between a normal car in pit versus a lapped car in pit,
    and as a result we cannot fix the lap number for the lapped car. After applying the above fix,
    we will get duplicated lap numbers for a lapped car if it pits after being lapped. We shift the
    lap number for the lapped car by 1 to get the correct lap number. See the below example: we
    have lap number 30, 31, 33, 33 and it should be 30, 31, 32, 33. We shift the first "33" to "32"
    to fix it.

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
    df['lap'] = df['lap'] - df['gap'].apply(lambda x: int(re.findall(r'\d+', x)[0]) if 'LAP' in x
                                                      else 0)
    df.reset_index(drop=False, inplace=True)
    df.sort_values(by=['driver_no', 'lap', 'index'], inplace=True)
    df.loc[(df['driver_no'] == df['driver_no'].shift(-1)) & (df['lap'] == df['lap'].shift(-1)),
           'lap'] -= 1
    df.loc[(df['driver_no'] == df['driver_no'].shift(1)) & (df['lap'] == df['lap'].shift(1) + 2),
           'lap'] -= 1
    del df['index']

    # TODO: Perez "retired and rejoined" in 2023 Japanese... Maybe just mechanically assign lap No.
    #       as 1, 2, 3, ...?
    return df


def to_timedelta(s: str) -> datetime.timedelta:
    """
    Covert a time string to a timedelta object, e.g. "1:32.190" -->
    datetime.timedelta(seconds=92, microseconds=190000)

    # TODO: move this to `utils.py`?
    """
    # Parse by ":" and "."
    n_colons = s.count(':')
    h, m, sec, ms = 0, 0, 0, 0
    match n_colons:
        case 1:  # "1:32.190"
            m, sec = s.split(':')
            sec, ms = sec.split('.')
        case 2:  # "1:32:19.190"
            warnings.warn(f'''got an unusual time: {s}. Assuming it's "hh:mm:ss.ms"''')
            h, m, sec = s.split(':')
            sec, ms = sec.split('.')
        case 0:  # "19.190"
            warnings.warn(f'''got an unusual time: {s}. Assuming it's "ss.ms"''')
            sec, ms = s.split('.')
        case _:  # Weird case
            raise ValueError(f'''got an unexpected time: {s}''')

    # Check if the time is valid
    assert 0 <= int(h) < 24, f'''hour should be in [0, 24), got {h} in {s}'''
    assert 0 <= int(m) < 60, f'''minute should be in [0, 60), got {m} in {s}'''
    assert 0 <= int(sec) < 60, f'''second should be in [0, 60), got {sec} in {s}'''
    assert 0 <= int(ms) < 1000, f'''millisecond should be in [0, 1000), got {ms} in {s}'''

    t = datetime.timedelta(hours=int(h), minutes=int(m), seconds=int(sec), milliseconds=int(ms))
    if t == datetime.timedelta(0):
        raise ValueError(f'''got an invalid time: {s}''')
    return t


def to_json(df: pd.DataFrame) -> list[dict]:
    """Convert the parsed lap time df. to a json obj. See jolpica/jolpica-f1#7"""

    # Hard code 2023 Abu Dhabi for now
    year = 2023
    round_no = 22
    session_type = 'R'

    # Convert string time time to timedelta, e.g. "1:32.190" -->
    df['time'] = df['time'].apply(to_timedelta)

    # Convert to json
    df['lap'] = df.apply(
        lambda x: Lap(number=x['lap'], position=x['position'], time=x['time']), axis=1
    )
    df = df.groupby('driver_no')[['lap']].agg(list).reset_index()
    df['session_entry'] = df['driver_no'].map(
        lambda x: SessionEntry(
            year=year,
            round=round_no,
            session=session_type,
            car_number=x
        )
    )
    del df['driver_no']
    lap_data = df.apply(
        lambda x: LapData(foreign_keys=x['session_entry'], objects=x['lap']).model_dump(),
        axis=1
    ).tolist()
    with open('laps.pkl', 'wb') as f:
        pickle.dump(lap_data, f)
    return lap_data


if __name__ == '__main__':
    pass
