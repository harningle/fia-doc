import json

import pytest

from fiadoc.parser import PitStopParser
from fiadoc.utils import download_pdf

race_list = [
    (
        '2023_14_ned_f1_r0_timing_racepitstopsummary_v01.pdf',
        2023,
        13,
        'race',
        '2023_13_race_pit_stop.json'
    ),
    (
        # Table very short (only one row)
        '2025_02_chn_f1_s0_timing_sprintpitstopsummary_v01.pdf',
        2025,
        2,
        'sprint',
        '2025_2_sprint_pit_stop.json'
    )
]


@pytest.fixture(params=race_list)
def prepare_pit_stop_data(request, tmp_path) -> tuple[list[dict], list[dict]]:
    # Download and parse quali. classification and lap times PDF
    url, year, round_no, session, expected = request.param
    download_pdf('https://www.fia.com/sites/default/files/' + url, tmp_path / 'pit_stop.pdf')
    parser = PitStopParser(tmp_path / 'pit_stop.pdf', year, round_no, session)

    data = parser.df.to_json()
    with open('fiadoc/tests/fixtures/' + expected, encoding='utf-8') as f:
        expected = json.load(f)

    # Sort data
    data.sort(key=lambda x: (x['foreign_keys']['car_number']))
    expected.sort(key=lambda x: (x['foreign_keys']['car_number']))
    for i in data:
        i['objects'].sort(key=lambda x: x['number'])
    for i in expected:
        i['objects'].sort(key=lambda x: x['number'])
    return data, expected


def test_parse_pit_stop(prepare_pit_stop_data):
    data, expected = prepare_pit_stop_data
    assert data == expected
