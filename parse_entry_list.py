import os
import pickle
import fitz 
import pandas as pd

from models.driver import Driver, RoundEntry
from models.foreign_key import Round


def parse_entry_list(file: str | os.PathLike) -> pd.DataFrame:
    """Parse the table from 'Entry List' PDF."""
    
    def parse_text_to_df(text):
        # Manually parse the text to extract table data
        lines = text.split('\n')
        data = []
        columns = ['No.', 'Driver', 'Nat', 'Team', 'Constructor']
        reserve_mode = False
        for line in lines:
            # Identify the start of the table
            if 'No.' in line and 'Driver' in line and 'Nat' in line and 'Team' in line and 'Constructor' in line:
                reserve_mode = False
                continue
            # Identify the start of reserve drivers
            elif 'In addition to the list' in line:
                reserve_mode = True
                continue

            parts = line.split()
            if len(parts) >= 5:
                no = parts[0]
                driver = ' '.join(parts[1:3])
                nat = parts[3]
                team = ' '.join(parts[4:-2])
                constructor = ' '.join(parts[-2:])
                role = 'reserve' if reserve_mode else 'permanent'
                data.append([no, driver, nat, team, constructor, role])
        
        df = pd.DataFrame(data, columns=columns + ['role'])
        return df
    
    doc = fitz.open(file)
    text = ""

    for page in doc:
        text += page.get_text()

    df = parse_text_to_df(text)
    
    return df

def to_json(df: pd.DataFrame):
    # Hard code 2023 Abu Dhabi for now
    year = 2023
    round_no = 22

    # To json
    print(df)
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
    return round_entry.dict()

if __name__ == '__main__':
    df = parse_entry_list('race_entry_list.pdf')
    round_entry = to_json(df)
    print(round_entry)
