import pandas as pd
from parse_race_fastest_laps import parse_race_fastest_laps, to_json


def test_parse_race_fastest_laps():
    df = parse_race_fastest_laps("fia_pdfs/race_fastest_laps.pdf")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20


def test_to_json():
    df = parse_race_fastest_laps("fia_pdfs/race_fastest_laps.pdf")
    fastest_lap_data = to_json(df)
    assert isinstance(fastest_lap_data, list)
    assert isinstance(fastest_lap_data[0], dict)
    assert len(fastest_lap_data) == 20
