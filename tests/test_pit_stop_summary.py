import pandas as pd
from parse_pit_stop_summary import parse_pit_stop_summary, to_json


def test_parse_pit_stop_summary():
    df = parse_pit_stop_summary("fia_pdfs/race_pit_stop_summary.pdf")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 37


def test_to_json():
    df = parse_pit_stop_summary("fia_pdfs/race_pit_stop_summary.pdf")
    pit_stop_data = to_json(df)
    assert isinstance(pit_stop_data, list)
    assert isinstance(pit_stop_data[0], dict)
    assert len(pit_stop_data) == 20
