import json

import pytest

from fiadoc.parser import RaceParser
from fiadoc.utils import download_pdf

race_list = [
    (
        'doc_66_-_2023_united_states_grand_prix_-_final_race_classification.pdf',
        '2023_19_usa_f1_r0_timing_racelapanalysis_v01.pdf',
        '2023_19_usa_f1_r0_timing_racehistorychart_v01.pdf',
        '2023_19_usa_f1_r0_timing_racelapchart_v01.pdf',
        2023,
        18,
        '2023_18_race_classification.json',
        '2023_18_race_lap_times.json'
    )
]


@pytest.fixture(params=race_list)
def prepare_race_data(request, tmp_path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    # Download and parse race classification and lap times PDFs
    url_classification, url_lap_analysis, url_history_chart, url_lap_chart, year, round_no, \
        expected_classification, expected_lap_times = request.param
    download_pdf('https://www.fia.com/sites/default/files/' + url_classification,
                 tmp_path / 'classification.pdf')
    download_pdf('https://www.fia.com/sites/default/files/' + url_lap_analysis,
                 tmp_path / 'lap_analysis.pdf')
    download_pdf('https://www.fia.com/sites/default/files/' + url_history_chart,
                 tmp_path / 'history_chart.pdf')
    download_pdf('https://www.fia.com/sites/default/files/' + url_lap_chart,
                 tmp_path / 'lap_chart.pdf')
    parser = RaceParser(tmp_path / 'classification.pdf', tmp_path / 'lap_analysis.pdf',
                        tmp_path / 'history_chart.pdf', tmp_path / 'lap_chart.pdf',
                        year, round_no, 'race')

    classification_data = parser.classification_df.to_json()
    lap_times_data = parser.lap_times_df.to_json()
    with open('fiadoc/tests/fixtures/' + expected_classification, encoding='utf-8') as f:
        expected_classification = json.load(f)
    with open('fiadoc/tests/fixtures/' + expected_lap_times, encoding='utf-8') as f:
        expected_lap_times = json.load(f)

    # Sort data
    classification_data.sort(key=lambda x: x['foreign_keys']['car_number'])
    expected_classification.sort(key=lambda x: x['foreign_keys']['car_number'])
    lap_times_data.sort(key=lambda x: x['foreign_keys']['car_number'])
    expected_lap_times.sort(key=lambda x: x['foreign_keys']['car_number'])
    for i in lap_times_data:
        i['objects'].sort(key=lambda x: x['number'])
    for i in expected_lap_times:
        i['objects'].sort(key=lambda x: x['number'])

    # TODO: currently manually tested against fastf1 lap times. The test data should be generated
    #       automatically later. Also need to manually add if the lap time is deleted and if the
    #       lap is fastest manually. Also need to add the laps where PDFs have data but fastf1
    #       doesn't
    return classification_data, lap_times_data, expected_classification, expected_lap_times


def test_parse_race(prepare_race_data):
    classification_data, lap_times_data, expected_classification, expected_lap_times \
        = prepare_race_data
    assert classification_data == expected_classification
    assert lap_times_data == expected_lap_times
