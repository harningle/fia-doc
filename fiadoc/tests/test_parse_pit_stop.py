import json
import os
import re
import warnings

import pytest

from fiadoc.parser import PitStopParser
from fiadoc.utils import download_pdf, sort_json

race_list = [
    (
        # Normal pit stop summary spanning over three pages
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

    return sort_json(data), sort_json(expected)


def test_parse_pit_stop(prepare_pit_stop_data):
    data, expected = prepare_pit_stop_data
    assert data == expected


@pytest.mark.full
@pytest.mark.parametrize('year, round_no',
                         [(2024, i) for i in range(1, 25)]
                         + [(2025, i) for i in range(1, 25)])
def test_parse_pit_stop_full(year: int, round_no: int):
    pdfs = [f'data/pdf/{i}' for i in os.listdir('data/pdf')
           if i.startswith(f'{year}_{round_no}_') and i.endswith('_pit_stop_summary.pdf')]
    if len(pdfs) == 0:
        warnings.warn(f"Pit stop PDF for {year} round {round_no} doesn't existed. Skipping")
        return
    for pdf in pdfs:
        session = re.findall(rf'{year}_{round_no}_(\w+)_pit', pdf)
        parser = PitStopParser(pdf, year, round_no, session[0])
        parser.df.to_json()
    return
