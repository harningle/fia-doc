# -*- coding: utf-8 -*-
import os
import pickle

import camelot
import fitz
import pandas as pd

from models.driver import Driver, RoundEntry
from models.foreign_key import Round


def parse_entry_list(file: str | os.PathLike) -> pd.DataFrame:
    """Parse the table from "Entry List" PDF.

    An example of `team` and `constructor` is "Alfa Romeo F1 Team Stake" and "Alfa Romeo Ferrari".

    `role` can be "permanent" or "reserve".

    See `notebook/demo.ipynb` for the detailed explanation of the table structure.

    :param file: Path to PDF file
    :return: A dataframe of [car No., driver name, nationality, team, constructor, role]
    """
    # Locate the table
    doc = fitz.open(file)
    page = doc[1]  # TODO: can have multiple pages
    w, h = page.bound()[2], page.bound()[3]
    car_no = page.search_for('No.')[0]
    top_left = (car_no.x0, car_no.y0)

    # Try to find the bottom of the table
    text_height = (car_no.y1 - car_no.y0) * 1.035  # TODO: line spacing seems to be roughly 1.035?
    top = car_no.y0 + text_height
    bottom = car_no.y1 + text_height  # Give it a little buffer
    no_left = car_no.x0
    no_right = car_no.x1
    while top < h:
        text = page.get_text('text', clip=(no_left, top, no_right, bottom))
        if text.strip():
            top += text_height
            bottom += text_height
        else:  # If find nothing, that's the end of the table
            break
    bottom_right = (w, top)

    # Flip the y-axis so we have the coordinates for `camelot`
    top_left = (int(top_left[0]), int(h - top_left[1]))
    bottom_right = (int(bottom_right[0]), int(h - bottom_right[1]) + 1)

    # Parse using `camelot`
    bbox = ','.join(map(str, top_left + bottom_right))
    tables = camelot.read_pdf(file, flavor='stream', pages='2', table_areas=[bbox], flag_size=True)
    df = tables[0].df

    # Clean up the superscript
    df.columns = df.iloc[0]
    df = df[df['No.'] != 'No.']
    df['No.'] = df['No.'].str.split('<s>').str[0]
    df['role'] = 'permanent'

    # Extract the table for reserve drivers
    top += text_height
    bottom += text_height
    while top < h:
        text = page.get_text('text', clip=(no_left, top, no_right, bottom))
        if text.strip():
            top += text_height
            bottom += text_height
        else:  # If find nothing, that's the start of the table
            break
    top_left = (no_left, bottom)
    top += text_height
    bottom += text_height
    while top < h:
        text = page.get_text('text', clip=(no_left, top, no_right, bottom))
        if text.strip():
            top += text_height
            bottom += text_height
        else:  # If find nothing, that's the end of the table
            break
    bottom_right = (w, top)
    top_left = (int(top_left[0]), int(h - top_left[1]))
    bottom_right = (int(bottom_right[0]), int(h - bottom_right[1]) + 1)
    bbox = ','.join(map(str, top_left + bottom_right))
    tables = camelot.read_pdf(file, flavor='stream', pages='2', table_areas=[bbox], flag_size=True)
    df_reserve = tables[0].df
    df_reserve['role'] = 'reserve'
    df_reserve.columns = df.columns
    df_reserve['No.'] = df_reserve['No.'].str.split('<s>').str[0]

    return pd.concat([df, df_reserve], ignore_index=True)


def to_json(df: pd.DataFrame):
    # Hard code 2023 Abu Dhabi for now
    year = 2023
    round_no = 22

    # To json
    df['driver'] = df.apply(
        lambda x: Driver(car_number=x['No.'], name=x['Driver'], team=x['Team'], role=x['role']),
        axis=1
    )
    drivers = df['driver'].tolist()
    round_entry = RoundEntry(
        foreign_keys=Round(year=year, round=round_no),
        objects=drivers
    )

    with open('entry_list.pkl', 'wb') as f:
        pickle.dump(round_entry.dict(), f)


if __name__ == '__main__':
    pass