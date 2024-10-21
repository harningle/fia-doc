# -*- coding: utf-8 -*-
"""
TODO: this is going to be merged with `parse_race_final_classification.py` for sure. The two  are
effectively the same thing
"""
import os
import pickle

import fitz
import pandas as pd

from models.foreign_key import SessionEntry
from models.classification import Classification, ClassificationData


def parse_sprint_final_classification(file: str | os.PathLike) -> pd.DataFrame:
    """Parse "Sprint Final Classification" PDF

    :param file: Path to PDF file
    :return: The output dataframe will be [driver No., laps completed, total time,
                                           finishing position, finishing status, fastest lap time,
                                           fastest lap speed, fastest lap No.]
    """
    # Find the page with "Sprint Final Classification"
    doc = fitz.open(file)
    for i in range(len(doc)):
        page = doc[i]
        found = page.search_for('Sprint Final Classification')
        if found:
            break

    # Width and height of the page
    w, _ = page.bound()[2], page.bound()[3]

    # Position of "Race Final Classification"
    y = found[0].y1

    # Position of "NOT CLASSIFIED" or "FASTEST LAP"
    not_classified = page.search_for('NOT CLASSIFIED')
    if not_classified:
        b = not_classified[0].y0
    else:
        b = page.search_for('FASTEST LAP')[0].y0

    # Table bounding box
    bbox = fitz.Rect(0, y, w, b)

    # Positions of table headers/column names
    pos = {}
    for col in ['NO', 'DRIVER', 'NAT', 'ENTRANT', 'LAPS', 'TIME', 'GAP', 'INT', 'KM/H', 'FASTEST',
                'ON', 'PTS']:
        pos[col] = {
            'left': page.search_for(col, clip=bbox)[0].x0,
            'right': page.search_for(col, clip=bbox)[0].x1
        }

    # Lines separating the columns
    aux_lines = [
        pos['NO']['left'],
        (pos['NO']['right'] + pos['DRIVER']['left']) / 2,
        pos['NAT']['left'],
        pos['NAT']['right'],
        pos['LAPS']['left'],
        pos['LAPS']['right'],
        (pos['TIME']['right'] + pos['GAP']['left']) / 2,
        (pos['GAP']['right'] + pos['INT']['left']) / 2,
        (pos['INT']['right'] + pos['KM/H']['left']) / 2,
        pos['FASTEST']['left'],
        pos['FASTEST']['right'],
        pos['PTS']['left']
    ]

    # Find the table below "Race Final Classification"
    df = page.find_tables(
        clip=fitz.Rect(pos['NO']['left'], y, w, b),
        strategy='lines',
        vertical_lines=aux_lines,
        snap_x_tolerance=pos['ON']['left'] - pos['FASTEST']['right']
    )[0].to_pandas()
    df = df[df['NO'] != '']  # May get some empty rows at the bottom

    # Clean a bit
    df.drop(columns=['DRIVER', 'NAT', 'ENTRANT', 'INT', 'KM/H', 'GAP'], inplace=True)
    df.rename(columns={
        'NO': 'driver_no',
        'LAPS': 'laps_completed',
        'TIME': 'time',
        'GAP': 'gap',
        'FASTEST': 'fastest_lap_time',
        'ON': 'fastest_lap_no',
        'Col11': 'points'  # TODO: this can be a bit fragile?
    }, inplace=True)
    df.driver_no = df.driver_no.astype(int)
    df.laps_completed = df.laps_completed.astype(int)
    df.time = pd.to_timedelta('00:' + df.time)  # TODO: not the best way to handle this
    # TODO: gap to the leader is to be cleaned later, so we can use it for cross validation
    # TODO: the fastest lap data is to be cleaned
    # df.fastest_lap_time = pd.to_timedelta(df.fastest_lap_time)  # TODO: this can be missing?
    # df.fastest_lap_no = df.fastest_lap_no.astype(int)  # TODO: this can be missing?
    df.replace({'points': {'': 0}}, inplace=True)  # Not all drivers get points, so missing -> 0
    df.points = df.points.astype(float)
    return df


def to_json(df: pd.DataFrame) -> dict:
    # Hard code 2023 Brazil for now
    year = 2023
    round_no = 20

    # Convert to json
    df['classification'] = df.apply(
        lambda x: ClassificationData(
            foreign_keys=SessionEntry(
                year=year,
                round=round_no,
                session='SR',
                car_number=x['driver_no']
            ),
            objects=[
                Classification(
                    points=x['points'],
                    time=x['time'],
                    laps_completed=x['laps_completed']
                )
            ]
        ).model_dump(),
        axis=1
    )

    # Dump to json
    classification_data = df['classification'].tolist()
    with open('sprint_final_classification.pkl', 'wb') as f:
        pickle.dump(classification_data, f)
    return classification_data


if __name__ == '__main__':
    pass
