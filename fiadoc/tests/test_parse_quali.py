import json
import os
from contextlib import nullcontext
from typing import Optional

import pytest

from fiadoc.parser import QualifyingParser
from fiadoc.utils import download_pdf

race_list = [
    (
        # Normal quali.
        '2024_22_usa_f1_q0_timing_qualifyingsessionprovisionalclassification_v01.pdf',
        '2024_22_usa_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        22,
        'quali',
        '2024_22_quali_classification.json',
        '2024_22_quali_lap_times.json',
        nullcontext()
    ),
    (
        # Title is image rather than string
        'doc_53_-_2024_chinese_grand_prix_-_final_qualifying_classification.pdf',
        '2024_05_chn_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        5,
        'quali',
        '2024_5_quali_classification.json',
        '2024_5_quali_lap_times.json',
        nullcontext()
    ),
    (
        # DNF drivers in quali.
        '2024_02_ksa_f1_q0_timing_qualifyingsessionprovisionalclassification_v01.pdf',
        '2024_02_ksa_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        2,
        'quali',
        '2024_2_quali_classification.json',
        '2024_2_quali_lap_times.json',
        nullcontext()
    ),
    (
        # DNF drivers in quali.
        '2024_21_bra_f1_sq0_timing_sprintqualifyingsessionprovisionalclassification_v01.pdf',
        '2024_21_bra_f1_sq0_timing_sprintqualifyingsessionlaptimes_v01.pdf',
        2024,
        21,
        'sprint_quali',
        '2024_21_sprint_quali_classification.json',
        '2024_21_sprint_quali_lap_times.json',
        nullcontext()
    ),
    (
        # Antonelli's name being long, which breaks the older parser
        # DNS drivers in quali.
        '2025_01_aus_f1_q0_timing_qualifyingsessionprovisionalclassification_v01.pdf',
        '2025_01_aus_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2025,
        1,
        'quali',
        '2025_1_quali_provisional_classification.json',
        '2025_1_quali_lap_times.json',
        nullcontext()
    ),
    (
        # DSQ drivers in quali.
        'https://www.fia.com/sites/default/files/decision-document/2024%20Monaco%20Grand%20Prix%20-%20Final%20Qualifying%20Classification.pdf',
        '2024_08_mon_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        8,
        'quali',
        '2024_8_quali_classification.json',
        '2024_8_quali_lap_times.json',
        nullcontext()
    ),
    (
        # No "POLE POSITION" in quali. classification
        'https://www.fia.com/sites/default/files/decision-document/2024%20Australian%20Grand%20Prix%20-%20Final%20Qualifying%20Classification.pdf',
        '2024_03_aus_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2024,
        3,
        'quali',
        '2024_3_quali_classification.json',
        '2024_3_quali_lap_times.json',
        nullcontext()
    ),
    (
        # Text is image in the PDF...
        'https://www.fia.com/system/files/decision-document/2025_australian_grand_prix_-_final_qualifying_classification.pdf',
        '2025_01_aus_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2025,
        1,
        'quali',
        '2025_1_quali_final_classification.json',
        '2025_1_quali_lap_times.json',
        nullcontext()
    ),
    (
        # Antonelli's name wrapped in two lines
        # DNF drivers in quali.
        'https://www.fia.com/system/files/decision-document/2025_chinese_grand_prix_-_final_sprint_qualifying_classification.pdf',
        '2025_02_chn_f1_sq0_timing_sprintqualifyingsessionlaptimes_v01.pdf',
        2025,
        2,
        'sprint_quali',
        '2025_2_sprint_quali_final_classification.json',
        '2025_2_sprint_quali_lap_times.json',
        nullcontext()
    ),
    (
        # DNQ drivers in quali. (#50)
        '2025_04_brn_f1_q0_timing_qualifyingsessionfinalclassification_v01.pdf',
        '2025_04_brn_f1_q0_timing_qualifyingsessionlaptimes_v01.pdf',
        2025,
        4,
        'quali',
        '2025_4_quali_classification.json',
        '2025_4_quali_lap_times.json',
        nullcontext()
    ),
    (
        # Without lap times PDF (#47)
        'https://www.fia.com/system/files/decision-document/2025_emilia_romagna_grand_prix_-_final_qualifying_classification.pdf',
        None,
        2025,
        7,
        'quali',
        '2025_7_quali_classification.json',
        '2025_7_quali_lap_times_lap_times_pdf_unavailable.json',
        pytest.warns(UserWarning, match='Lap times PDF is missing')
    ),
    (
        # Lap times are incorrectly matched with quali. sessions (#51)
        'https://www.fia.com/system/files/decision-document/2025_emilia_romagna_grand_prix_-_final_qualifying_classification.pdf',
        '2025_07_ita_f1_q0_timing_qualifyingsessionlaptimes_v01_0.pdf',
        2025,
        7,
        'quali',
        '2025_7_quali_classification.json',
        '2025_7_quali_lap_times_fallback.json',
        nullcontext()
    )
]


@pytest.fixture(params=race_list)
def prepare_quali_data(request, tmp_path) \
        -> tuple[list[dict], Optional[list[dict]], list[dict], Optional[list[dict]]]:
    # Download and parse quali. classification and lap times PDF
    url_classification, url_lap_time, year, round_no, session, expected_classification, \
        expected_lap_times, context = request.param
    if 'https://' not in url_classification:  # TODO: clean this up
        url_classification = 'https://www.fia.com/sites/default/files/' + url_classification
    download_pdf(url_classification, tmp_path / 'classification.pdf')
    if url_lap_time:  # Whether lap times PDF is provided
        download_pdf('https://www.fia.com/sites/default/files/' + url_lap_time,
                     tmp_path / 'lap_times.pdf')
        parser = QualifyingParser(tmp_path / 'classification.pdf', tmp_path / 'lap_times.pdf',
                                  year, round_no, session)
    else:
        parser = QualifyingParser(tmp_path / 'classification.pdf', None, year, round_no, session)

    with context:
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
    for i in lap_times_data:
        i['objects'].sort(key=lambda x: x['number'])
    expected_lap_times.sort(
        key=lambda x: (x['foreign_keys']['session'], x['foreign_keys']['car_number'])
    )
    for i in expected_lap_times:
        i['objects'].sort(key=lambda x: x['number'])

    # TODO: currently manually tested against fastf1 lap times. The test data should be generated
    #       automatically later. Also need to manually add if the lap time is deleted and if the
    #       lap is fastest manually. Also need to add the laps where PDFs have data but fastf1
    #       doesn't
    return classification_data, lap_times_data, expected_classification, expected_lap_times


def test_parse_quali(prepare_quali_data):
    classification_data, lap_times_data, expected_classification, expected_lap_times = \
        prepare_quali_data
    assert classification_data == expected_classification
    assert lap_times_data == expected_lap_times

    """
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
    """
    return


@pytest.mark.full
@pytest.mark.parametrize('year, round_no', [(2024, i) for i in range(1, 25)])
def test_parse_quali_full(year: int, round_no: int):
    classification_pdf = f'data/pdf/{year}_{round_no}_quali_provisional_classification.pdf'
    lap_times_pdf = f'data/pdf/{year}_{round_no}_quali_lap_times.pdf'
    if not os.path.exists(classification_pdf):
        raise FileNotFoundError(f"Quali. classification PDF for {year} round {round_no} doesn't "
                                f"exist")
    if not os.path.exists(lap_times_pdf):
        raise FileNotFoundError(f"Quali. lap times PDF for {year} round {round_no} doesn't exist")
    parser = QualifyingParser(classification_pdf, lap_times_pdf, year, round_no, 'quali')
    parser.classification_df.to_json()
    parser.lap_times_df.to_json()
    return


@pytest.mark.full
@pytest.mark.parametrize('year, round_no', [(2024, i) for i in [5, 6, 19, 21, 23]])
# @pytest.mark.parametrize('year, round_no', [(2024, i) for i in [5, 6, 11, 19, 21, 23]])
# Skip 2024 Austrian as no sprint lap time PDF available on FIA website
# TODO: see #47. Need to add it back or even include it in the usual test above
def test_parse_sprint_quali(year: int, round_no: int):
    classification_pdf = f'data/pdf/{year}_{round_no}_sprint_quali_provisional_classification.pdf'
    lap_times_pdf = f'data/pdf/{year}_{round_no}_sprint_quali_lap_times.pdf'
    if not os.path.exists(classification_pdf):
        raise FileNotFoundError(f"Sprint quali. classification PDF for {year} round {round_no} "
                                f"doesn't exist")
    if not os.path.exists(lap_times_pdf):
        raise FileNotFoundError(f"Sprint quali. lap times PDF for {year} round {round_no} doesn't "
                                f"exist")
    parser = QualifyingParser(classification_pdf, lap_times_pdf, year, round_no, 'sprint_quali')
    parser.classification_df.to_json()
    parser.lap_times_df.to_json()
    return
