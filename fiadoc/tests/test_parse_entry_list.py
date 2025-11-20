import json
import os
import warnings
from contextlib import nullcontext
from tempfile import TemporaryDirectory

import pytest
import requests_mock

from fiadoc.drivers import BASE_URL
from fiadoc.parser import EntryListParser
from fiadoc.utils import download_pdf, sort_json

race_list = [
    (
        # Normal entry list
        '2024%20S%C3%A3o%20Paulo%20Grand%20Prix%20-%20Revised%20Entry%20List.pdf',
        2024,
        21,
        '2024_21_entry_list.json',
        nullcontext()
    ),
    (
        # Have multiple reserve drivers who are not in the 20 usual drivers (#49)
        '2024%20Mexico%20City%20Grand%20Prix%20-%20Entry%20List.pdf',
        2024,
        20,
        '2024_20_entry_list.json',
        pytest.warns(UserWarning, match='New drivers found in entry list PDF')
    ),
    (
        # Have only one reserve driver (#55)
        '2024%20Japanese%20Grand%20Prix%20-%20Entry%20List.pdf',
        2024,
        4,
        '2024_4_entry_list.json',
        pytest.warns(UserWarning, match='New drivers found in entry list PDF')
    ),
    (
        # Have a driver (Ricciardo) incorrectly indicated as having a reserve driver (looks like
        # FIA copy-paste error from previous race) (#23)
        '2024%20Chinese%20Grand%20Prix%20-%20Entry%20List.pdf',
        2024,
        5,
        '2024_5_entry_list.json',
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
        pytest.warns(UserWarning, match='New drivers found in entry list PDF')
    ),
    (
        # Car No. superscript shown as regular text w/ smaller font size, rather than a proper
        # superscript (#48)
        '2025_bahrain_grand_prix_-_entry_list.pdf',
        2025,
        4,
        '2025_4_entry_list.json',
        pytest.warns(UserWarning, match='New drivers found in entry list PDF')
    ),
    (
        # Two and only two reserve drivers (#55)
        '2025_spanish_grand_prix_-_entry_list.pdf',
        2025,
        9,
        '2025_9_entry_list.json',
        pytest.warns(UserWarning, match='New drivers found in entry list PDF')
    )
]
# Not going to test year 2023 for entry list, as their PDF format is different, and we are not
# interested in retrospectively parsing old entry list PDFs

os.environ['FIADOC_CACHE_DIR'] = TemporaryDirectory().name


@pytest.fixture
def jolpica_mock(requests_mock: requests_mock.Mocker):
    """Mock Jolpica API endpoints"""
    requests_mock.real_http = True  # Allow unmocked requests to pass through

    with open('fiadoc/tests/fixtures/cached_drivers.json', 'r', encoding='utf8') as f:
        cached_drivers = json.loads(f.read())
    n_cached_drivers = str(len(cached_drivers))

    requests_mock.get(
        f'{BASE_URL}/drivers/?format=json&limit=1',
        json={'MRData': {'total': n_cached_drivers}}
    )
    requests_mock.get(
        f'{BASE_URL}/drivers',
        json={
            'MRData': {
                'total': n_cached_drivers,
                'DriverTable': {
                    'Drivers': cached_drivers
                }
            }
        }
    )


@pytest.fixture(params=race_list)
def prepare_entry_list_data(request, tmp_path, jolpica_mock) -> tuple[list[dict], list[dict]]:
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
    return sort_json(data), sort_json(expected_data)


def test_parse_entry_list(prepare_entry_list_data):
    data, expected_data = prepare_entry_list_data
    assert data == expected_data


@pytest.mark.full
@pytest.mark.parametrize('year, round_no',
                         [(2024, i) for i in range(1, 25)]
                         + [(2025, i) for i in range(1, 25)])
def test_parse_entry_list_full(year: int, round_no: int):
    pdfs = [f'data/pdf/{i}' for i in os.listdir('data/pdf')
           if i.startswith(f'{year}_{round_no}_') and i.endswith('_entry_list.pdf')]
    if len(pdfs) == 0:
        warnings.warn(f"Entry list PDF for {year} round {round_no} doesn't existed. Skipping")
        return
    for pdf in pdfs:  # May have the usual entry list and a revised entry list etc.
        parser = EntryListParser(pdf, year, round_no)
        parser.df.to_json()
    return
