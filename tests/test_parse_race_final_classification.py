import pandas as pd
import pytest

from fiadoc.parser import ClassificationParser


def test_parse_final_classification():
    df = parse_race_final_classification_page('race_final_classification.pdf')
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 20
