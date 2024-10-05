import pandas as pd
from parse_race_final_classification import parse_race_final_classification_page


def test_parse_race_final_classification_page():
    df = parse_race_final_classification_page("fia_pdfs/race_final_classification.pdf")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
