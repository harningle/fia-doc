import pandas as pd
from parse_race_lap_chart import parse_race_lap_chart, to_json


def test_parse_race_lap_chart():
    df = parse_race_lap_chart("fia_pdfs/race_lap_chart.pdf")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1157  # 20 cars * 58 laps - 3 lapped cars


def test_to_json():
    df = parse_race_lap_chart("fia_pdfs/race_lap_chart.pdf")
    lap_data = to_json(df)
    assert isinstance(lap_data, list)
    assert isinstance(lap_data[0], dict)
    assert len(lap_data) == 20
