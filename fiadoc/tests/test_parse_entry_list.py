from contextlib import nullcontext
import json

import pytest

from fiadoc.parser import EntryListParser
from fiadoc.utils import download_pdf

race_list = [
    (
        '2024%20S%C3%A3o%20Paulo%20Grand%20Prix%20-%20Revised%20Entry%20List.pdf',
        2024,
        21,
        '2024_21_entry_list.json',
        nullcontext()
    ),
    (
        '2024%20Belgian%20Grand%20Prix%20-%20Entry%20List.pdf',
        2024,
        14,
        '2024_14_entry_list.json',
        nullcontext()
    ),
    (
        '2024%20Mexico%20City%20Grand%20Prix%20-%20Entry%20List.pdf',
        2024,
        20,
        '2024_20_entry_list.json',
        # multiple reserve drivers that must be skipped on export to json
        # because they are not in the driver mapping
        pytest.warns(UserWarning, match='Error when parsing driver')
    ),
    (
        # handle gracefully if driver is not found in the driver mapping
        # (e.g. reserve driver not in the mapping) and warn user
        '2024%20Japanese%20Grand%20Prix%20-%20Entry%20List.pdf',
        2024,
        4,
        '2024_04_entry_list.json',
        pytest.warns(UserWarning, match='Error when parsing driver Ayumu')
    ),
    (
        # RIC incorrectly indicated as having a reserve driver here
        # (looks like copy-paste error from previous race) -> handle gracefully
        '2024%20Chinese%20Grand%20Prix%20-%20Entry%20List.pdf',
        2024,
        5,
        '2024_05_entry_list.json',
        pytest.warns(UserWarning, match='Ricciardo is indicated as')
    ),
    (
        # Weird PDF page margin (#33)
        '2025_australian_grand_prix_-_entry_list_corrected_.pdf',
        2025,
        1,
        '2025_1_entry_list.json',
        nullcontext()
    ),
    (
        # Weird PDF page margin (#33)
        '2025_japanese_grand_prix_-_entry_list.pdf',
        2025,
        3,
        '2025_3_entry_list.json',
        pytest.warns(UserWarning, match='Error when parsing driver Ryo Hirakawa')
    )
]
# Not going to test year 2023 for entry list, as the PDF format changed, and we are not interested
# in retrospectively parsing old entry list PDFs


@pytest.fixture(params=race_list)
def prepare_entry_list_data(request, tmp_path) -> tuple[list[dict], list[dict]]:
    # Download and parse entry list PDF
    url, year, round_no, expected, context = request.param
    if year <= 2024:
        base_url = 'https://www.fia.com/sites/default/files/decision-document/'
    else:
        base_url = 'https://www.fia.com/system/files/decision-document/'
    download_pdf(base_url + url,
                 tmp_path / 'entry_list.pdf')

    with context:
        parser = EntryListParser(tmp_path / 'entry_list.pdf', year, round_no)
        data = parser.df.to_json()

    # Sort by car No. for both json for easier comparison
    with open('fiadoc/tests/fixtures/' + expected, encoding='utf-8') as f:
        expected_data = json.load(f)
    data.sort(key=lambda x: x['objects'][0]['car_number'])
    expected_data.sort(key=lambda x: x['objects'][0]['car_number'])
    return data, expected_data


def test_parse_entry_list(prepare_entry_list_data):
    data, expected_data = prepare_entry_list_data
    assert data == expected_data
