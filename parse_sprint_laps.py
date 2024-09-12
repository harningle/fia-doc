# -*- coding: utf-8 -*-
import datetime
import os
import pickle
import re
import warnings

import fitz
import pandas as pd

from models.foreign_key import SessionEntry
from models.lap import Lap, LapData


def parse_sprint_lap_analysis_page(page: fitz.Page) -> pd.DataFrame:
    """Parse a page in "Sprint Lap Analysis" PDF"""
    w = page.bound()[2]                               # Page width. Shared by three drivers
    t = page.search_for('Sprint Lap Analysis')[0].y1  # "Race Lap Analysis" at the top of the page
    b = page.search_for('Page')[0].y0                 # "Page" at the bottom right of the page

    # Find the tables inside the left/middle/right part of the page
    """
    On the page, the drivers and tables are laid out in the following way:
    
    | driver 1              | driver 2              | driver 3              |
    | table 1.1 | table 1.2 | table 2.1 | table 2.2 | table 3.1 | table 3.2 |
    | driver 4              | driver 5              | driver 6              |
    | table 4.1 | table 4.2 | table 5.1 | table 5.2 | table 6.1 | table 6.2 |
    | ...
    
    Horizontally, there are three parts: left, middle, and right. Each part contains a driver's
    name and his tables (or perhaps only one if the race only has very few laps). Sometimes (if we
    have a lot of laps so the tables are tall), one part only has one driver vertically. In other
    times (only a few laps, as in the sprint, so the tables are short), one part can have multiple
    drivers vertically.
    
    TODO: we assume pymupdf returns tables in the order of 1.1, 1.2, 2.1, 2.2 below. Should check
    if this is always the case.
    """
    df = []
    for i in range(3):
        # Find all tables in this part
        l = i * w / 3
        r = (i + 1) * w / 3
        tabs = page.find_tables(clip=fitz.Rect(l, t, r, b), strategy='lines')
        if not tabs.tables:  # E.g., 20 drivers, 3 parts, 20 % 3 != 0, so will not all parts have
            continue         # driver. TODO: this can be tested, e.g., in this example we should
                             # only has one part with no driver
        # Find the driver's name and link tables to the driver
        assert tabs[0].to_pandas().iloc[0, 0] == '1'  # The 1st table's 1st cell should be lap #. 1
        driver_tabs = []  # Store the tables for the driver being processed currently
        for tab in tabs:
            if tab.to_pandas().iloc[0, 0] == '1':   # The 1st table for a driver has lap #. 1
                # If we already have some tables and now find a new driver, then the tables belong
                # to the previous driver
                if driver_tabs:
                    for j in driver_tabs:
                        j['driver'] = name
                        j['car_no'] = car_no
                        assert (j.Col1 == '').all()  # TODO: check this
                        del j['Col1']
                        j.rename(columns={'LAP': 'lap', 'TIME': 'time'}, inplace=True)
                        j.lap = j.lap.astype(int)
                        df.append(j)
                    driver_tabs = []
                h = tab.header.bbox[1]  # Top of the table
                name = page.get_text('block', clip=fitz.Rect(l, h - 30, r, h)).strip()
                car_no, name = name.split('\n')
            driver_tabs.append(tab.to_pandas())

        # Process the last driver in this part. His tables won't be appended in the loop above
        for j in driver_tabs:
            j['driver'] = name
            j['car_no'] = car_no
            assert (j.Col1 == '').all()  # TODO: check this
            del j['Col1']
            j.rename(columns={'LAP': 'lap', 'TIME': 'time'}, inplace=True)
            j.lap = j.lap.astype(int)
            df.append(j)

    return pd.concat(df, ignore_index=True)


def parse_sprint_lap_analysis(file: str | os.PathLike) -> pd.DataFrame:
    """Parse "Sprint Lap Analysis" PDF"""
    doc = fitz.open(file)
    df = []
    for page in doc:
        df.append(parse_sprint_lap_analysis_page(page))
    return pd.concat(df, ignore_index=True)


def parse_sprint_history_chart_page(page: fitz.Page) -> pd.DataFrame:
    """
    Get the table(s) from a given page in "Sprint History Chart" PDF. There are multiple tables in
    a page, each of which correspond to a lap No. We concat all tables into one single dataframe

    See `notebook/demo.ipynb` for the detailed explanation of the table structure.

    :param page: A `fitz.Page` object
    :return: A dataframe of [driver No., lap No., gap to leader, lap time]

    TODO: probably use better type hint using pandera later
    TODO: merge this with race lap parsing script
    """

    # Get the position of "Lap x"
    t = page.search_for('Sprint History Chart')[0].y1
    b = page.search_for('TIME')[0].y1
    headers = page.search_for('Lap', clip=(0, t, W, b))

    # Iterate through the tables for each lap
    tables = []
    for i, lap in enumerate(headers):
        """
        The left boundary of the table is the leftmost of the "Lap x" text, and the right boundary
        is the leftmost of the next "Lap x" text. If it's the last lap, i.e. no next table, then
        the right boundary can be determined by left boundary plus table width, which is roughly
        one-fifth of the page width. We add 5% extra buffer to the right boundary
        """
        l = lap.x0
        r = headers[i + 1].x0 if i + 1 < len(headers) else (l + W / 5) * 1.05
        temp = page.find_tables(clip=fitz.Rect(l, t, r, H),
                                strategy='lines',
                                add_lines=[((l, 0), (l, H))])[0].to_pandas()

        # Three columns: "LAP x", "GAP", "TIME". "LAP x" is the column for driver No. So add a new
        # column for lap No. with value "x", and rename the columns
        lap_no = int(temp.columns[0].split(' ')[1])
        temp.columns = ['driver_no', 'gap', 'time']
        temp['lap'] = lap_no
        temp = temp[temp['driver_no'] != '']  # Sometimes we will get one additional empty row

        # The row order/index is meaningful: it's the order/positions of the cars
        # TODO: is this true for all cases? E.g. retirements?
        temp.reset_index(drop=False, names=['position'], inplace=True)
        temp['position'] += 1  # 1-indexed
        tables.append(temp)
    return pd.concat(tables, ignore_index=True)


def parse_sprint_history_chart(file: str | os.PathLike[str]) -> pd.DataFrame:
    """
    Parse "Sprint History Chart" PDF

    :param file: Path to PDF file
    :return: The output dataframe will be [driver No., lap No., gap to leader, lap time]
    """
    # Get page width and height
    doc = fitz.open(file)
    page = doc[0]
    global W, H
    W = page.bound()[2]
    H = page.bound()[3]

    # Parse all pages
    df = pd.concat([parse_sprint_history_chart_page(page) for page in doc], ignore_index=True)

    # Clean up
    # TODO: check notes in `parse_race_history_chart.py`
    df['lap'] = df['lap'] - df['gap'].apply(
        lambda x: int(re.findall(r'\d+', x)[0]) if 'LAP' in x else 0
    )
    df.reset_index(drop=False, inplace=True)
    df.sort_values(by=['driver_no', 'lap', 'index'], inplace=True)
    df.loc[(df['driver_no'] == df['driver_no'].shift(-1)) & (df['lap'] == df['lap'].shift(-1)),
           'lap'] -= 1
    df.loc[(df['driver_no'] == df['driver_no'].shift(1)) & (df['lap'] == df['lap'].shift(1) + 2),
           'lap'] -= 1
    del df['index']
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

    # Hard code 2023 Brazil for now
    year = 2023
    round_no = 20
    session_type = 'SR'

    # Convert string time time to timedelta
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
    with open('sprint_laps.pkl', 'wb') as f:
        pickle.dump(lap_data, f)
    return lap_data
