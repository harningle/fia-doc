from contextlib import nullcontext
import json

import pytest

from fiadoc.parser import PracticeParser
from fiadoc.utils import download_pdf, sort_json

race_list = [
    (
        # 0: Normal practice
        '2025_11_aut_f1_p1_timing_firstpracticesessionclassification_v01.pdf',
        '2025_11_aut_f1_p1_timing_firstpracticesessionlaptimes_v01.pdf',
        2025,
        11,
        'fp1',
        '2025_11_fp1_classification.json',
        '2025_11_fp1_lap_times.json',
        nullcontext()
    ),
    (
        # 1: Ocon fails to set a valid time, i.e. has a few laps but all deleted/invalid
        '2024_17_aze_f1_p1_timing_firstpracticesessionclassification_v01.pdf',
        '2024_17_aze_f1_p1_timing_firstpracticesessionlaptimes_v01.pdf',
        2024,
        17,
        'fp1',
        '2024_17_fp1_classification.json',
        '2024_17_fp1_lap_times.json',
        nullcontext()
    ),
    (
        # 2: Lap times PDF unavailable
        '2025_dutch_grand_prix_-_fp3_classification.pdf',
        None,
        2025,
        15,
        'fp3',
        '2025_15_fp3_classification.json',
        '2025_15_fp3_lap_times_fallback.json',
        pytest.warns(UserWarning, match='Lap times PDF is missing')
    ),
    (
        # 3: Weird one-driver-a-row layout lap times PDF...
        '2025_01_aus_f1_p1_timing_firstpracticesessionclassification_v01.pdf',
        '2025_01_aus_f1_p1_timing_firstpracticesessionlaptimes_v01.pdf',
        2025,
        1,
        'fp1',
        '2025_1_fp1_classification.json',
        '2025_1_fp1_lap_times.json',
        nullcontext()
    ),
    (
        # 4: A driver has no lap at all, i.e. DNS
        '2026_01_aus_f1_p1_timing_firstpracticesessionclassification_v01.pdf',
        '2026_01_aus_f1_p1_timing_firstpracticesessionlaptimes_v01.pdf',
        2026,
        1,
        'fp1',
        '2026_1_fp1_classification.json',
        '2026_1_fp1_lap_times.json',
        nullcontext()
    ),
    (
        # 5: Perez has no valid lap, and the lap times PDF doesn't mark pit laps correctly (#77)
        # This is different from Ocon's case in test case 1, where the lap times PDF does correctly
        # mark all pit laps
        '2026_01_aus_f1_p2_timing_secondpracticesessionclassification_v01.pdf',
        '2026_01_aus_f1_p2_timing_secondpracticesessionlaptimes_v01.pdf',
        2026,
        1,
        'fp2',
        '2026_1_fp2_classification.json',
        '2026_1_fp2_lap_times.json',
        nullcontext()
    ),
    (
        # 6: different PDF style in decision doc. (#78)
        'https://www.fia.com/system/files/decision-document/2025_emilia_romagna_grand_prix_-_p1_classification.pdf',
        None,
        2025,
        7,
        'fp1',
        '2025_7_fp1_classification.json',
        '2025_7_fp1_lap_times_fallback.json',
        pytest.warns(UserWarning, match='Lap times PDF is missing')
    )
]


@pytest.fixture(params=race_list)
def prepare_fp_data(request, tmp_path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    # Download and parse race classification and lap times PDFs
    url_classification, url_lap_time, year, round_no, session, expected_classification, \
        expected_lap_times, context = request.param
    if url_classification.startswith('http'):
        download_pdf(url_classification, tmp_path / 'classification.pdf')
    else:
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
        lap_times_data = parser.lap_times_df.to_json()
    with open('fiadoc/tests/fixtures/' + expected_classification, encoding='utf-8') as f:
        expected_classification = json.load(f)
    if expected_lap_times:
        with open('fiadoc/tests/fixtures/' + expected_lap_times, encoding='utf-8') as f:
            expected_lap_times = json.load(f)
    else:
        expected_lap_times = None

    return (sort_json(classification_data),     sort_json(lap_times_data),
            sort_json(expected_classification), sort_json(expected_lap_times))


def test_parse_fp(prepare_fp_data):
    classification_data, lap_times_data, expected_classification, expected_lap_times \
        = prepare_fp_data
    assert classification_data == expected_classification
    assert lap_times_data == expected_lap_times
