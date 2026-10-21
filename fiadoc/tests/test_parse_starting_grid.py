import json

import pytest

from fiadoc.parser import StartingGridParser
from fiadoc.tests._warnings import assert_warnings
from fiadoc.utils import download_pdf, sort_json

race_list = [
    (
        # 0: Pole on the right. Some drivers start from the pit lane, and one of them has no
        #    quali. time. Some drivers have penalties
        'https://www.fia.com/system/files/decision-document/2026_singapore_grand_prix_-_final_starting_grid.pdf',
        2026,
        17,
        'race',
        '2026_17_race_starting_grid.json',
        assert_warnings()
    ),
    (
        # 1: Sprint. Pole on the left. Some drivers on the grid have no quali. time. Zhou has
        #    family name first
        'https://www.fia.com/sites/default/files/decision-document/2024%20Austrian%20Grand%20Prix%20-%20Final%20Sprint%20Starting%20Grid.pdf',
        2024,
        11,
        'sprint',
        '2024_11_sprint_starting_grid.json',
        assert_warnings()
    ),
    (
        # 2: Page size is US Letter rather than A4, so everything is positioned differently
        'https://www.fia.com/system/files/decision-document/2025_belgian_grand_prix_-_final_sprint_starting_grid.pdf',
        2025,
        13,
        'sprint',
        '2025_13_sprint_starting_grid.json',
        assert_warnings()
    )
]


@pytest.fixture(params=race_list)
def prepare_starting_grid_data(request, tmp_path) -> tuple[list[dict], list[dict]]:
    # Download and parse starting grid PDF
    url, year, round_no, session, expected, context = request.param
    download_pdf(url, tmp_path / 'starting_grid.pdf')
    parser = StartingGridParser(tmp_path / 'starting_grid.pdf', year, round_no, session)

    with context:
        data = parser.df.to_json()
    with open('fiadoc/tests/fixtures/' + expected, encoding='utf-8') as f:
        expected = json.load(f)

    return sort_json(data), sort_json(expected)


def test_parse_starting_grid(prepare_starting_grid_data):
    data, expected = prepare_starting_grid_data
    assert data == expected
