# -*- coding: utf-8 -*-
import os
import pickle

import fitz
import pandas as pd

from models.lap import Lap, LapData
from models.foreign_key import SessionEntry


def parse_race_lap_chart_page(page: fitz.Page, page_width: float) -> pd.DataFrame:
    """Get the table from a given page in "Race Lap Chart" PDF

    :param page: A `fitz.Page` object
    :return: A dataframe of [lap No., position, driver No.]

    TODO: probably use better type hint using pandera later
    """

    # Get the position of "POS" and "Page", between which the table is located vertically
    # TODO: Probably need to use some other text as reference point. If the race name has "POS" in
    #       it, then the current method will fail
    page_top = page.search_for('POS')[0].y0
    page_bottom = page.search_for('Page')[0].y0

    df = page.find_tables(clip=fitz.Rect(0, page_top, page_width, page_bottom), strategy='text')[0].to_pandas()

    """
    The parsing is not always successful. We may have one of the following situations:
    
    1. we do have the columns correct
    2. the "POS" column somehow is separated into "P" and "OS" column
    
    Additionally, we many have an empty row as the first row. See `notebook/demo.ipynb` for the
    detailed explanation
    """

    # Clean the lap No. col.
    df.replace('', None, inplace=True)
    df.dropna(how='all', inplace=True)
    if 'POS' in df.columns:
        df = df[df['POS'] != 'GRID']  # Probably need this row later as the "actual" starting grid
        df['POS'] = df['POS'].str.extract(r'(\d+)')[0].astype(int)
    elif 'P' in df.columns and 'OS' in df.columns:
        del df['P']
        df.rename(columns={'OS': 'POS'}, inplace=True)
    else:
        raise ValueError('Failed to parse the table. Check the PDF file')
    df.rename(columns={'POS': 'lap'}, inplace=True)
    return df


def parse_race_lap_chart(file: str | os.PathLike[str]) -> pd.DataFrame:
    """
    Parse "Race Lap Chart" PDF

    :param file: Path to PDF file
    :return: The output dataframe will be [lap No., position, driver No.]
    """
    # Get page width and height
    doc = fitz.open(file)
    page = doc[0]
    page_width = page.bound()[2]

    # Parse all pages
    tables = []
    for page in doc:
        tables.append(parse_race_lap_chart_page(page, page_width))
    df = pd.concat(tables, ignore_index=True)

    # Reshape the table to long format, i.e. to lap-position level
    df.set_index('lap', inplace=True)
    df = df.stack().reset_index()
    df.columns = ['lap', 'position', 'driver_no']
    for col in ['lap', 'position', 'driver_no']:
        df[col] = df[col].astype(int)
    return df


def to_json(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the parsed lap time df. to a json obj. See jolpica/jolpica-f1#7"""

    # Hard code 2023 Abu Dhabi for now
    year = 2023
    round_no = 22
    session_type = 'R'

    # Convert to json
    df['lap'] = df.apply(lambda x: Lap(number=x['lap'], position=x['position'], time=0.0),
                         axis=1)
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
