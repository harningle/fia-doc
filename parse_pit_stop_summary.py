# -*- coding: utf-8 -*-
import os
import pickle

import fitz
import pandas as pd

from models.pit_stop import PitStop, PitStopData
from models.foreign_key import SessionEntry


def parse_pit_stop_summary(file: str | os.PathLike[str]) -> pd.DataFrame:
    """Parse the table from "Pit Stop Summary" PDF

    See `notebook/demo.ipynb` for the detailed explanation of the table structure.

    :param file: Path to PDF file
    :return: A dataframe of [driver No., lap No., local time of the stop, pit stop No., duration]
    """

    doc = fitz.open(file)
    page = doc[0]
    # TODO: definitely have PDFs containing multiple pages

    # Get the position of the table
    t = page.search_for('DRIVER')[0].y0      # "DRIVER" gives the top of the table
    w, h = page.bound()[2], page.bound()[3]  # Page width and height
    bbox = fitz.Rect(0, t, w, h)

    # Parse
    df = page.find_tables(clip=bbox, strategy='lines')[0].to_pandas()

    # Clean up the table
    df.dropna(subset=['NO'], inplace=True)  # Drop empty rows, if any
    df = df[df['NO'] != '']
    df = df[['NO', 'LAP', 'TIME OF DAY', 'STOP', 'DURATION']].reset_index(drop=True)
    df.rename(columns={
        'NO': 'driver_no',
        'LAP': 'lap',
        'TIME OF DAY': 'local_time',
        'STOP': 'no',
        'DURATION': 'duration'
    }, inplace=True)
    return df


def to_json(df: pd.DataFrame) -> list[dict]:
    """Convert the parsed df. to a json obj. See jolpica/jolpica-f1#7"""

    # Hard code 2023 Abu Dhabi for now
    year = 2023
    round_no = 22
    session_type = 'R'

    # Pit stop duration to timedelta
    df['duration'] = df['duration'].apply(lambda x: pd.to_timedelta(float(x), unit='s'))

    # Convert to json
    df['pit_stop'] = df.apply(lambda x: PitStop(
        lap=x['lap'], number=x['no'],  duration=x['duration'], local_timestamp=x['local_time']
    ), axis=1)
    df = df.groupby('driver_no')[['pit_stop']].agg(list).reset_index()
    df['session_entry'] = df['driver_no'].map(
        lambda x: SessionEntry(
            year=year,
            round=round_no,
            type=session_type,
            car_number=x
        )
    )
    del df['driver_no']
    pit_stop_data = df.apply(
        lambda x: PitStopData(foreign_keys=x['session_entry'], objects=x['pit_stop']).model_dump(),
        axis=1
    ).tolist()
    with open('pit_stops.pkl', 'wb') as f:
        pickle.dump(pit_stop_data, f)
    return pit_stop_data


if __name__ == '__main__':
    pass
