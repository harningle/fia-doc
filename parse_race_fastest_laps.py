# -*- coding: utf-8 -*-
import os

import fitz
import pandas as pd


def parse_race_fastest_laps(file: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse the table from "Race Fastest Laps" PDF

    See `notebook/demo.ipynb` for the detailed explanation of the table structure.

    :param file: Path to PDF file
    :return: A dataframe of [driver No., lap No., fastest lap rank]
    """

    # This PDF should only have one page
    doc = fitz.open(file)
    assert len(doc) == 1, f'More than one page found in the Race Fastest Laps PDF: {file}'
    page = doc[0]

    # Get the position of the table
    h, w = page.bound()[3], page.bound()[2]         # Height and width of the page
    t = page.search_for('Race Fastest Laps')[0].y1  # "Race Fastest Lap"
    b = page.search_for('TIME OF DAY')[0].y1        # "TIME OF DAY"
    bbox = fitz.Rect(0, t, w, b)

    # Col. positions
    pos = {}
    for col in ['NO', 'DRIVER', 'NAT', 'ENTRANT', 'TIME', 'ON', 'GAP', 'INT', 'KM/H',
                'TIME OF DAY']:
        pos[col] = {
            'left': page.search_for(col, clip=bbox)[0].x0,
            'right': page.search_for(col, clip=bbox)[0].x1
        }
    aux_lines = [
        pos['NO']['left'],
        pos['DRIVER']['left'],
        pos['NAT']['left'],
        pos['NAT']['right'],
        pos['TIME']['left'] - (pos['TIME']['right'] - pos['TIME']['left']),
        (pos['TIME']['right'] + pos['ON']['left']) / 2,
        (pos['ON']['right'] + pos['GAP']['left']) / 2,
        (pos['GAP']['right'] + pos['INT']['left']) / 2,
        (pos['INT']['right'] + pos['KM/H']['left']) / 2,
        pos['TIME OF DAY']['left'],
        pos['TIME OF DAY']['right']
    ]

    # Parse the table
    df = page.find_tables(
        clip=fitz.Rect(pos['NO']['left'], t, pos['TIME OF DAY']['right'], h),
        strategy='lines',
        vertical_lines=aux_lines,
        snap_x_tolerance=pos['ENTRANT']['left'] - pos['NAT']['right']  # TODO: this is very fragile
    )[0].to_pandas()

    # Clean up the table
    df.dropna(subset=['NO'], inplace=True)  # May get some empty rows
    df = df[df['NO'] != '']
    df = df[['NO', 'ON']].reset_index(drop=True)
    df.rename(columns={'NO': 'driver_no', 'ON': 'lap'}, inplace=True)
    df['fastest_lap_rank'] = range(1, len(df) + 1)
    return df


if __name__ == '__main__':
    pass
