"""Classification and lap properties must reuse their shared parsing dependency."""
import json
from pathlib import Path

import pytest

from fiadoc.parser import RaceParser
from fiadoc.tests.test_parse_race import race_list
from fiadoc.utils import download_pdf, sort_json


@pytest.mark.parametrize('properties', [
    ['classification_df', 'lap_times_df'],
    ['lap_times_df', 'classification_df'],
])
def test_race_properties_parse_classification_once(properties, tmp_path, monkeypatch):
    case = next(c for c in race_list if '2026_13_race_classification.json' in c)
    paths = []
    for i, url in enumerate(case[:5]):
        if url is None:
            paths.append(None)
            continue
        resolved = (url if url.startswith('https://') else
                    'https://www.fia.com/sites/default/files/' + url)
        path = tmp_path / f'{i}.pdf'
        download_pdf(resolved, path)
        paths.append(path)
    original = RaceParser._parse_classification
    calls = 0

    def counted(parser):
        nonlocal calls
        calls += 1
        return original(parser)

    monkeypatch.setattr(RaceParser, '_parse_classification', counted)
    parser = RaceParser(*paths, *case[5:8])
    output = {name: getattr(parser, name).to_json() for name in properties}
    assert calls == 1
    for name, fixture in zip(['classification_df', 'lap_times_df'], case[8:10]):
        expected = json.loads((Path('fiadoc/tests/fixtures') / fixture).read_text())
        assert sort_json(output[name]) == sort_json(expected)
