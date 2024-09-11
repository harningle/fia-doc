# -*- coding: utf-8 -*-
"""TODO: should be refactored/merged with `parse_sprint_laps.py` later"""
import os
import pickle

import fitz
import pandas as pd

from models.foreign_key import SessionEntry


def parse_sprint_shootout_final_classification(file: str | os.PathLike) -> pd.DataFrame:
    """Parse "Sprint Shootout Session Final Classification" PDF"""
    # Find the page with "Sprint Shootout Session Final Classification"
    doc = fitz.open(file)
    found = None
    for i in range(len(doc)):
        page = doc[i]
        found = page.search_for('Sprint Shootout Session Final Classification')
        if found:
            break
    if found is None:
        raise ValueError(f'not able to find sprint shootout result in `{file}`')

    # Width and height of the page
    w, h = page.bound()[2], page.bound()[3]

    # y-position of "Sprint Shootout Final Classification"
    y = found[0].y1

    # y-position of "NOT CLASSIFIED - " or "POLE POSITION LAP"
    not_classified = page.search_for('NOT CLASSIFIED - ')
    b = None
    if not_classified:
        b = not_classified[0].y0
    else:
        b = page.search_for('POLE POSITION LAP')[0].y0
    if b is None:
        raise ValueError(f'not able to find the bottom of sprint shootout result in `{file}`')

    # Table bounding box
    bbox = fitz.Rect(0, y, w, b)

    # Dist. between "NAT" and "ENTRANT"
    nat = page.search_for('NAT')[0]
    entrant = page.search_for('ENTRANT')[0]
    snap_x_tolerance = (entrant.x0 - nat.x1) * 1.2  # 20% buffer

    # Parse
    df = page.find_tables(clip=bbox, snap_x_tolerance=snap_x_tolerance)[0].to_pandas()
    df.columns = ['_', 'NO', 'DRIVER', 'NAT', 'ENTRANT', 'SQ1', 'SQ1_LAPS', 'SQ1_%', 'SQ1_TIME',
                  'SQ2', 'SQ2_LAPS', 'SQ2_TIME', 'SQ3', 'SQ3_LAPS', 'SQ3_TIME']
    df.drop(columns=['_', 'NAT', 'ENTRANT'], inplace=True)
    df = df[df['NO'] != '']
    df.dropna(subset='NO', inplace=True)

    # Empty string to `None`, e.g. `SQ2_LAPS` is missing if the driver is in bottom 5 in SQ1
    df.replace('', None, inplace=True)
    return df


def get_strikeout_text(page: fitz.Page) -> list[tuple[fitz.Rect, str]]:
    """Get the strikeout text and their locations in the page

    TODO: move this to `utils.py` later maybe

    See https://stackoverflow.com/a/74582342/12867291.

    :param page: fitz.Page object
    :return: A list of tuples, where each tuple is the bbox and the strikeout text
    """
    # Get all strikeout lines
    lines = []
    paths = page.get_drawings()  # Strikeout lines are in fact vector graphics. To be more precise,
    for path in paths:           # they are rectangles with very small height
        for item in path['items']:
            if item[0] == 're':  # If a graphic is a rectangle, check if a horizontal line
                rect = item[1]
                if rect.width <= 2 * rect.height or rect.height > 1:
                    continue     # If width-to-height ratio is less than 2, it's not a line
                lines.append(rect)

    # Get all texts on this page
    # TODO: the O(n^2) here can probably be optimised later
    words = page.get_text('words')
    strikeout = []
    for rect in lines:
        for w in words:
            text_rect = fitz.Rect(w[:4])  # bbox of the word
            text = w[4]  # word text
            if text_rect.intersects(rect):
                strikeout.append((rect, text))
    return strikeout


def parse_sprint_shootout_lap_times_page(page: fitz.Page) -> pd.DataFrame:
    """Parse a page in "Sprint Shootout Session Lap Times" PDF

    This is the most complicated parsing. See `notebook/demo.ipynb` for more details.
    """
    # Page width. All tables are horizontally bounded in [0, `w`]
    w = page.bound()[2]

    # Positions of "NO" and "TIME". They are the top of each table. One driver may have multiple
    # tables starting from roughly the same top position
    no_time_pos = page.search_for('NO TIME')
    ys = [i.y1 for i in no_time_pos]
    ys.sort()                      # Sort these "NO"'s from top to bottom
    top_pos = [ys[0] - 1]          # -1 to give a little buffer at the top
    for y in ys[1:]:
        if y - top_pos[-1] > 10:   # Many "No"'s are at the same height, and we only need those at
            top_pos.append(y - 1)  # different heights. If there is 10px gap, we take it as diff.

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

    # Find the tables located between `top_pos` and `bottom_pos`
    df = []
    for row in range(len(top_pos)):
        for col in range(3):  # Three drivers in one row
            # Find the driver name, which is located immediately above the table
            driver = page.get_text(
                'block',
                clip=fitz.Rect(
                    col * w / 3,
                    top_pos[row] - 30,
                    (col + 1) * w / 3,
                    top_pos[row] - 10
                )
            ).strip()
            if not driver:  # In the very last row may not have all three drivers
                continue
            car_no, driver = driver.split(maxsplit=1)

            # Find tables in the bounding box of driver i's table
            tabs = page.find_tables(
                clip=fitz.Rect(col * w / 3, top_pos[row], (col + 1) * w / 3, bottom_pos[row]),
                strategy='lines'
            )  # TODO: I think this may fail if a driver only has one lap and crashes...

            # Check if any lap times is strikeout
            # TODO: the O(n^2) here can probably be optimised later
            # TODO: we should introduce the colour check as well: the strikeout texts are in grey,
            #       while normal texts are black
            strikeout_texts = get_strikeout_text(page)
            temp = []
            lap_time_deleted = []
            for tab in tabs:
                for r in tab.rows:
                    assert len(r.cells) == 3  # Should have lap, if pit, and lap time columns
                    cell = r.cells[2]  # Lap time cell (the last) can be strikeout
                    is_strikeout = False
                    bbox = fitz.Rect(cell)
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
    return pd.concat(df, ignore_index=True)


def parse_sprint_shootout_lap_times(file: str | os.PathLike) -> pd.DataFrame:
    """Parse "Sprint Shootout Session Lap Times" PDF"""
    # Get page width and height
    doc = fitz.open(file)

    # Parse all pages
    tables = []
    for page in doc:
        tables.append(parse_sprint_shootout_lap_times_page(page))
    df = pd.concat(tables, ignore_index=True)
    df['lap_no'] = df['lap_no'].astype(int)
    return df


def parse_date(d: str) -> pd.Timedelta:
    """Parse date string to datetime

    There can be two possible input formats:

    1. hh:mm:ss, e.g. 18:05:42. This is simply the local calendar time
    2. mm:ss.SSS, e.g. 1:24.160. This is the lap time

    TODO: maybe combine all time parsing functions into `utils.py` later
    """
    n_colon = d.count(':')
    if n_colon == 2:
        h, m, s = d.split(':')
        return pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s))
    elif n_colon == 1:
        m, s = d.split(':')
        s, ms = s.split('.')
        return pd.Timedelta(minutes=int(m), seconds=int(s), milliseconds=int(ms))
    else:
        raise ValueError(f'unknown date format: {d}')


def parse_sprint_shootout(final_path: str | os.PathLike, lap_times_path: str | os.PathLike) \
        -> pd.DataFrame:
    """TODO: probably need to refactor this later... To tedious now"""
    # Assign session to lap No., e.g. lap No. 7 is Q2, using final classification
    df = parse_sprint_shootout_lap_times(lap_times_path)
    classification = parse_sprint_shootout_final_classification(final_path)
    classification['SQ1_LAPS'] = classification['SQ1_LAPS'].astype(float)
    classification['SQ2_LAPS'] = classification['SQ2_LAPS'].astype(float) \
                                 + classification['SQ1_LAPS']
    df = df.merge(classification[['NO', 'SQ1_LAPS', 'SQ2_LAPS']],
                  left_on='car_no', right_on='NO', how='left')
    # TODO: check the laps in final classification and in lap times PDF. Why Sargent has 6 laps in
    #       final classification but only 5 laps in lap times PDF???
    # TODO: I don't remember what happened last year in Brazil. Why Ricciardo had only one pit stop
    #       but made into SQ3? Shouldn't be SQ1 and box and SQ2 and box, so at least two pit stops?
    # TODO: should check here if all drivers are merged. All unmerged ones should be not classified
    del df['NO']
    df['Q'] = 1
    df.loc[df['lap_no'] > df['SQ1_LAPS'], 'Q'] = 2
    df.loc[df['lap_no'] > df['SQ2_LAPS'], 'Q'] = 3
    # TODO: should check the lap before the first Q2 and Q3 lap is pit lap. Or is it? Crashed?
    del df['SQ1_LAPS'], df['SQ2_LAPS']

    # Find which lap is the fastest lap, also using final classification
    df['calendar_time'] = df['lap_time'].apply(parse_date)
    df['calendar_time'] = df.groupby('car_no')['calendar_time'].cumsum()
    df['is_fastest_lap'] = False
    for q in [1, 2, 3]:
        # Round to the floor
        df['temp'] = df['calendar_time'].apply(lambda x: str(x).split('.')[0].split(' ')[-1])
        df = df.merge(classification[['NO', f'SQ{q}_TIME']],
                      left_on=['car_no', 'temp'],
                      right_on=['NO', f'SQ{q}_TIME'],
                      how='left')
        del df['NO']
        # Plus one to the floor, i.e. allow one second error in the merge
        df['temp'] = df['calendar_time'].apply(
            lambda x: str(x + pd.Timedelta(seconds=1)).split('.')[0].split(' ')[-1]
        )
        df = df.merge(classification[['NO', f'SQ{q}_TIME']],
                      left_on=['car_no', 'temp'],
                      right_on=['NO', f'SQ{q}_TIME'],
                      how='left',
                      suffixes=('', '_y'))
        del df['NO'], df['temp']
        df.fillna({f'SQ{q}_TIME': df[f'SQ{q}_TIME_y']}, inplace=True)
        del df[f'SQ{q}_TIME_y']
        # Check if all drivers in the final classification are merged
        temp = classification[['NO', f'SQ{q}_TIME']].merge(
            df[df[f'SQ{q}_TIME'].notnull()][['car_no']],
            left_on='NO',
            right_on='car_no',
            indicator=True
        )
        temp.dropna(subset=f'SQ{q}_TIME', inplace=True)
        assert (temp['_merge'] == 'both').all()
        df.loc[df[f'SQ{q}_TIME'].notnull(), 'is_fastest_lap'] = True
        del df[f'SQ{q}_TIME']

    # Clean up
    df['pit'] = (df['pit'] == 'P').astype(bool)
    return df

