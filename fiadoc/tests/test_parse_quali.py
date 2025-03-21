import json

import pytest

from fiadoc.parser import QualifyingParser
from fiadoc.utils import download_pdf

race_list = [
    (
        '2024_22_usa_f1_q0_timing_qualifyingsessionprovisionalclassification_v01.pdf',
        '2024_22_usa_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        22,
        'quali',
        '2024_22_quali_provisional_classification.json',
        '2024_22_quali_lap_times.json'
    ),
    (
        'doc_20_-_2023_united_states_grand_prix_-_final_qualifying_classification.pdf',
        '2023_19_usa_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2023,
        18,
        'quali',
        '2023_18_quali_classification.json',
        '2023_18_quali_lap_times.json'
    ),
    (
        'doc_32_-_2023_united_states_grand_prix_-_final_sprint_shootout_classification.pdf',
        '2023_19_usa_f1_sq0_timing_sprintshootoutsessionlaptimes_v01.pdf',
        2023,
        18,
        'sprint_quali',
        '2023_18_sprint_quali_classification.json',
        '2023_18_sprint_quali_lap_times.json'
    ),
    (
        # Page header is an image instead of text
        'doc_53_-_2024_chinese_grand_prix_-_final_qualifying_classification.pdf',
        '2024_05_chn_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        5,
        'quali',
        '2024_5_quali_classification.json',
        '2024_5_quali_lap_times.json'
    ),
    (
        '2024_02_ksa_f1_q0_timing_qualifyingsessionprovisionalclassification_v01.pdf',
        '2024_02_ksa_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        2,
        'quali',
        '2024_2_quali_classification.json',
        '2024_2_quali_lap_times.json'
    ),
    (
        '2024_21_bra_f1_sq0_timing_sprintqualifyingsessionprovisionalclassification_v01.pdf',
        '2024_21_bra_f1_sq0_timing_sprintqualifyingsessionlaptimes_v01.pdf',
        2024,
        21,
        'sprint_quali',
        '2024_21_sprint_quali_classification.json',
        '2024_21_sprint_quali_lap_times.json'
    ),
    (
        'doc_37_-_2023_spanish_grand_prix_-_final_qualifying_classification.pdf',
        '2023_08_esp_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2023,
        7,
        'quali',
        '2023_7_quali_classification.json',
        '2023_7_quali_lap_times.json'
    ),
    (
        # New quali. parser due to Antonelli's name being long
        '2025_01_aus_f1_q0_timing_qualifyingsessionprovisionalclassification_v01.pdf',
        '2025_01_aus_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2025,
        1,
        'quali',
        '2025_1_quali_provisional_classification.json',
        '2025_1_quali_lap_times.json'
    ),
    (
        # DSQ drivers in quali.
        'https://www.fia.com/sites/default/files/decision-document/2024%20Monaco%20Grand%20Prix%20-%20Final%20Qualifying%20Classification.pdf',
        '2024_08_mon_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        8,
        'quali',
        '2024_8_quali_classification.json',
        '2024_8_quali_lap_times.json'
    ),
    (
        # No "POLE POSITION" in quali. classification
        'https://www.fia.com/sites/default/files/decision-document/2024%20Australian%20Grand%20Prix%20-%20Final%20Qualifying%20Classification.pdf',
        '2024_03_aus_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        3,
        'quali',
        '2024_3_quali_classification.json',
        '2024_3_quali_lap_times.json'
    ),
    (
        # Text is image in the PDF...
        'https://www.fia.com/system/files/decision-document/2025_australian_grand_prix_-_final_qualifying_classification.pdf',
        '2025_01_aus_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2025,
        1,
        'quali',
        '2025_1_quali_final_classification.json',
        '2025_1_quali_lap_times.json'
    )
]


@pytest.fixture(params=race_list)
def prepare_quali_data(request, tmp_path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    # Download and parse quali. classification and lap times PDF
    url_classification, url_lap_time, year, round_no, session, expected_classification, \
        expected_lap_times = request.param
    if 'https://' not in url_classification:  # TODO: clean this up
        url_classification = 'https://www.fia.com/sites/default/files/' + url_classification
    download_pdf(url_classification, tmp_path / 'classification.pdf')
    download_pdf('https://www.fia.com/sites/default/files/' + url_lap_time,
                 tmp_path / 'lap_times.pdf')
    parser = QualifyingParser(tmp_path / 'classification.pdf', tmp_path / 'lap_times.pdf',
                              year, round_no, session)

    classification_data = parser.classification_df.to_json()
    lap_times_data = parser.lap_times_df.to_json()
    with open('fiadoc/tests/fixtures/' + expected_classification, encoding='utf-8') as f:
        expected_classification = json.load(f)
    with open('fiadoc/tests/fixtures/' + expected_lap_times, encoding='utf-8') as f:
        expected_lap_times = json.load(f)

    # Sort data
    classification_data.sort(
        key=lambda x: (x['foreign_keys']['session'], x['foreign_keys']['car_number'])
    )
    expected_classification.sort(
        key=lambda x: (x['foreign_keys']['session'], x['foreign_keys']['car_number'])
    )
    lap_times_data.sort(
        key=lambda x: (x['foreign_keys']['session'], x['foreign_keys']['car_number'])
    )
    expected_lap_times.sort(
        key=lambda x: (x['foreign_keys']['session'], x['foreign_keys']['car_number'])
    )
    for i in lap_times_data:
        i['objects'].sort(key=lambda x: x['number'])
    for i in expected_lap_times:
        i['objects'].sort(key=lambda x: x['number'])

    # TODO: currently manually tested against fastf1 lap times. The test data should be generated
    #       automatically later. Also need to manually add if the lap time is deleted and if the
    #       lap is fastest manually. Also need to add the laps where PDFs have data but fastf1
    #       doesn't
    return classification_data, lap_times_data, expected_classification, expected_lap_times


def test_parse_quali(prepare_quali_data):
    classification_data, lap_times_data, expected_classification, expected_lap_times \
        = prepare_quali_data
    assert classification_data == expected_classification

    # TODO: need to test against fastf1 in a better and more readable way
    for i in lap_times_data:
        driver = i['foreign_keys']['car_number']
        session = i['foreign_keys']['session']
        laps = i['objects']
        for j in expected_lap_times:
            if j['foreign_keys']['car_number'] == driver \
                    and j['foreign_keys']['session'] == session:
                expected_laps = j['objects']
                for lap in laps:
                    # Here we allow the lap to be missing in fastf1 data
                    for expected_lap in expected_laps:
                        if lap['number'] == expected_lap['number']:
                            assert lap['time'] == expected_lap['time'], \
                                f"Driver {driver}'s lap {lap['number']} in {session} time " \
                                f"doesn't match with fastf1: {lap['time']['milliseconds']} vs " \
                                f"{expected_lap['time']['milliseconds']}"
                            break

    for i in expected_lap_times:
        driver = i['foreign_keys']['car_number']
        session = i['foreign_keys']['session']
        expected_laps = i['objects']
        for j in lap_times_data:
            if j['foreign_keys']['car_number'] == driver \
                    and j['foreign_keys']['session'] == session:
                laps = i['objects']
                for expected_lap in expected_laps:
                    found = False
                    for lap in laps:
                        if lap['number'] == expected_lap['number']:
                            found = True
                            break
                    # But here any lap available in fastf1 data should be in PDF as well
                    assert found, f"Driver {driver}'s lap {expected_lap['number']} in {session} " \
                                  f"in fastf1 not found in PDF"
                    assert lap['time'] == expected_lap['time'], \
                        f"Driver {driver}'s lap {expected_lap['number']} in {session} time " \
                        f"doesn't match with fastf1: {lap['time']['milliseconds']} vs " \
                        f"{expected_lap['time']['milliseconds']}"
