import json

import pytest

from fiadoc.parser import EntryListParser
from fiadoc.utils import download_pdf

race_list = [
    (
        '2024%20S%C3%A3o%20Paulo%20Grand%20Prix%20-%20Revised%20Entry%20List.pdf',
        2024,
        21,
        '2024_21_entry_list.json'
    ),
    (
        '2024%20Belgian%20Grand%20Prix%20-%20Entry%20List.pdf',
        2024,
        14,
        '2024_14_entry_list.json'
    )
]


@pytest.fixture(params=race_list)
def prepare_entry_list_data(request, tmp_path) -> tuple[list[dict], list[dict]]:
    # Download and parse entry list PDF
    url, year, round_no, expected = request.param
    download_pdf('https://www.fia.com/sites/default/files/decision-document/' + url,
                 tmp_path / 'entry_list.pdf')
    parser = EntryListParser(tmp_path / 'entry_list.pdf', year, round_no)

    # Sort by car No. for both json for easier comparison
    data = parser.df.to_json()
    with open('fiadoc/tests/fixtures/' + expected) as f:
        expected_data = json.load(f)
    data.sort(key=lambda x: x['objects'][0]['car_number'])
    expected_data.sort(key=lambda x: x['objects'][0]['car_number'])
    return data, expected_data


def test_parse_entry_list(prepare_entry_list_data):
    data, expected_data = prepare_entry_list_data
    assert data == expected_data
