from contextlib import nullcontext
import json

import pytest

from fiadoc.parser import PracticeParser
from fiadoc.utils import download_pdf

race_list = [
    (
        # Normal practice
        '2025_11_aut_f1_p1_timing_firstpracticesessionclassification_v01.pdf',
        '2025_11_aut_f1_p1_timing_firstpracticesessionlaptimes_v01.pdf',
        2025,
        11,
        'fp1',
        '2025_11_fp1_classification.json',
        None,
        nullcontext()
    ),
]


@pytest.fixture(params=race_list)
def prepare_fp_data(request, tmp_path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    # Download and parse race classification and lap times PDFs
    url_classification, url_lap_time, year, round_no, session, expected_classification, \
        expected_lap_times, context = request.param
    download_pdf('https://www.fia.com/sites/default/files/' + url_classification,
                     tmp_path / 'classification.pdf')
    if url_lap_time:
        download_pdf('https://www.fia.com/sites/default/files/' + url_lap_time,
                     tmp_path / 'lap_times.pdf')
        parser = PracticeParser(tmp_path / 'classification.pdf', tmp_path / 'lap_times.pdf',
                                year, round_no, session)
    else:
        parser = PracticeParser(tmp_path / 'classification.pdf', None, year, round_no, session)

    with context:
        classification_data = parser.classification_df.to_json()
        lap_times_data = None
        # lap_times_data = parser.lap_times_df.to_json()
    with open('fiadoc/tests/fixtures/' + expected_classification, encoding='utf-8') as f:
        expected_classification = json.load(f)
    if expected_lap_times:
        with open('fiadoc/tests/fixtures/' + expected_lap_times, encoding='utf-8') as f:
            expected_lap_times = json.load(f)
    else:
        expected_lap_times = None

    # Sort data
    classification_data.sort(key=lambda x: x['foreign_keys']['car_number'])
    expected_classification.sort(key=lambda x: x['foreign_keys']['car_number'])
    if lap_times_data:
        lap_times_data.sort(key=lambda x: x['foreign_keys']['car_number'])
        for i in lap_times_data:
            i['objects'].sort(key=lambda x: x['number'])
    if expected_lap_times:
        expected_lap_times.sort(key=lambda x: x['foreign_keys']['car_number'])
        for i in expected_lap_times:
            i['objects'].sort(key=lambda x: x['number'])
    return classification_data, lap_times_data, expected_classification, expected_lap_times


def test_parse_fp(prepare_fp_data):
    classification_data, lap_times_data, expected_classification, expected_lap_times \
        = prepare_fp_data
    assert classification_data == expected_classification
    # assert lap_times_data == expected_lap_times
