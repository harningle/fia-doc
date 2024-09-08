import pandas as pd
from parse_entry_list import parse_entry_list, to_json


def test_parse_entry_list():
    df = parse_entry_list("fia_pdfs/race_entry_list.pdf")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24

    df = parse_entry_list("fia_pdfs/race_entry_list.pdf")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 24


def test_to_json():
    df = parse_entry_list("fia_pdfs/race_entry_list.pdf")
    entry_list_data = to_json(df)
    print(entry_list_data)
    assert isinstance(entry_list_data, dict)
    assert isinstance(entry_list_data["objects"], list)
    assert len(entry_list_data["objects"]) == 24
