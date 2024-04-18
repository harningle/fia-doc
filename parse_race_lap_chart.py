# -*- coding: utf-8 -*-
import os
import pickle

import fitz
import pandas as pd

from models.lap import Lap, LapData, SessionEntry

W: int  # Page width


def parse_race_lap_chart_page(page: fitz.Page) -> pd.DataFrame:
    """
    Get the table from a given page in "Race Lap Chart" PDF

    :param page: A `fitz.Page` object
    :return: A dataframe of [lap No., position, driver No.]

    TODO: probably use better type hint using pandera later
    """

    # Get the position of "POS" and "Page", between which the table is located vertically
    # TODO: Probably need to use some other text as reference point. If the race name has "POS" in
    #       it, then the current method will fail
    t = page.search_for('POS')[0].y0
    b = page.search_for('Page')[0].y0

    df = page.find_tables(clip=fitz.Rect(0, t, W, b), strategy='text')[0].to_pandas()

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
    global W
    W = page.bound()[2]

    # Parse all pages
    tables = []
    for page in doc:
        tables.append(parse_race_lap_chart_page(page))
    df = pd.concat(tables, ignore_index=True)

    # Reshape the table to long format, i.e. to lap-position level
    df.set_index('lap', inplace=True)
    df = df.stack().reset_index()
    df.columns = ['lap', 'position', 'driver_no']
    for col in ['lap', 'position', 'driver_no']:
        df[col] = df[col].astype(int)
    return df


def to_json(df: pd.DataFrame):
    """Convert the parsed lap time df. to a json obj. See jolpica/jolpica-f1#7"""

    # Hard code 2023 Abu Dhabi for now
    year = 2023
    round_no = 22

    # Hard code the team and driver
    driver_team = {
        1: 'Red Bull',
        11: 'Red Bull',
        16: 'Ferrari',
        55: 'Ferrari',
        63: 'Mercedes',
        44: 'Mercedes',
        31: 'Alpine',
        10: 'Alpine',
        81: 'McLaren',
        4: 'McLaren',
        77: 'Alfa Romeo',
        24: 'Alfa Romeo',
        18: 'Aston Martin',
        14: 'Aston Martin',
        20: 'Haas',
        27: 'Haas',
        3: 'AlphaTauri',
        22: 'AlphaTauri',
        23: 'Williams',
        2: 'Williams',
    }
    driver_name = {
        1: 'Max Verstappen',
        11: 'Sergio Perez',
        16: 'Charles Leclerc',
        55: 'Carlos Sainz',
        63: 'George Russell',
        44: 'Lewis Hamilton',
        31: 'Esteban Ocon',
        10: 'Pierre Gasly',
        81: 'Lando Norris',
        4: 'Lando Norris',
        77: 'Valtteri Bottas',
        24: 'Zhou Guanyu',
        18: 'Lance Stroll',
        14: 'Fernando Alonso',
        20: 'Kevin Magnussen',
        27: 'Nico Hulkenberg',
        3: 'Daniel Ricciardo',
        22: 'Yuki Tsunoda',
        23: 'Alexandre Albon',
        2: 'Logan Sargeant',
    }

    # Convert to json
    df['lap'] = df.apply(lambda x: Lap(number=x['lap'], position=x['position']), axis=1)
    df = df.groupby('driver_no')[['lap']].agg(list).reset_index()
    df['session_entry'] = df['driver_no'].map(
        lambda x: SessionEntry(
            round_number=round_no,
            season_year=year,
            team=driver_team[x],
            driver=driver_name[x]
        )
    )
    del df['driver_no']
    lap_data = df.apply(
        lambda x: LapData(foreign_keys=x['session_entry'], data=x['lap']).dict(),
        axis=1
    ).tolist()
    with open('laps.pkl', 'wb') as f:
        pickle.dump(lap_data, f)
    pass


if __name__ == '__main__':
    pass
