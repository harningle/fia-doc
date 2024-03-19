# -*- coding: utf-8 -*-
import os
import re

import fitz
import pandas as pd

W = None  # Page width and height
H = None


def parse_race_history_chart_page(page: fitz.Page) -> pd.DataFrame:
    """
    Get the table(s) from a given page in "Race History Chart" PDF. There are multiple tables in a
    page, each of which correspond to a lap No. We concat all tables into one single dataframe

    See `notebook/demo.ipynb` for the detailed explanation of the table structure

    :param page: A `fitz.Page` object
    :return: A dataframe of [driver No., lap No., gap to leader, lap time]

    TODO: probably use better type hint using pandera later
    """

    # Get the position of "Lap x"
    t = page.search_for('Race History Chart')[0].y1
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
    global W, H
    W = page.bound()[2]
    H = page.bound()[3]

    # Parse all pages
    tables = []
    for page in doc:
        tables.append(parse_race_history_chart_page(page))
    df = pd.concat(tables, ignore_index=True)

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
    #       col. as 1, 2, 3, ...?
    return df


if __name__ == '__main__':
    pass
