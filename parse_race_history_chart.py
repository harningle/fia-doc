# -*- coding: utf-8 -*-
import os

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
    return pd.concat(tables, ignore_index=True)


if __name__ == '__main__':
    pass
