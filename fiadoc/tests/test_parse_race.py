from contextlib import nullcontext
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
        'race',
        '2023_18_race_classification.json',
        '2023_18_race_lap_times.json',
        nullcontext()
    ),
    (
        'doc_43_-_2023_united_states_grand_prix_-_final_sprint_classification.pdf',
        '2023_19_usa_f1_s0_timing_sprintlapanalysis_v01.pdf',
        '2023_19_usa_f1_s0_timing_sprinthistorychart_v01.pdf',
        '2023_19_usa_f1_s0_timing_sprintlapchart_v01.pdf',
        2023,
        18,
        'sprint',
        '2023_18_sprint_classification.json',
        '2023_18_sprint_lap_times.json',
        nullcontext()
    ),
    (
        # test an event without unclassified drivers
        '2024_10_esp_f1_r0_timing_raceprovisionalclassification_v01_1.pdf',
        '2024_10_esp_f1_r0_timing_racelapanalysis_v01_1.pdf',
        '2024_10_esp_f1_r0_timing_racehistorychart_v01_1.pdf',
        '2024_10_esp_f1_r0_timing_racelapchart_v01_1.pdf',
        2024,
        10,
        'race',
        '2024_10_race_classification.json',
        '2024_10_race_lap_times.json',
        nullcontext()
    ),
    (
        'doc_50_-_2024_monaco_grand_prix_-_provisional_race_classification.pdf',
        '2024_08_mon_f1_r0_timing_racelapanalysis_v01.pdf',
        '2024_08_mon_f1_r0_timing_racehistorychart_v01.pdf',
        '2024_08_mon_f1_r0_timing_racelapchart_v01.pdf',
        2024,
        8,
        'race',
        '2024_8_race_classification.json',
        '2024_8_race_lap_times.json',
        nullcontext()
    ),
    (
        'doc_76_-_2023_austrian_grand_prix_-_final_race_classification.pdf',
        '2023_10_aut_f1_r0_timing_racelapanalysis_v01.pdf',
        '2023_10_aut_f1_r0_timing_racehistorychart_v01.pdf',
        '2023_10_aut_f1_r0_timing_racelapchart_v01.pdf',
        2023,
        9,
        'race',
        '2023_9_race_classification.json',
        '2023_9_race_lap_times.json',
        nullcontext()
    ),
    (
        # DNF but classified, e.g., crashed in final lap, but finished 90%+ of the race
        # (jolpica/jolpica-f1#223, jolpica/jolpica-f1#246)
        'https://www.fia.com/system/files/decision-document/2025_canadian_grand_prix_-_final_race_classification.pdf',
        '2025_10_can_f1_r0_timing_racelapanalysis_v01.pdf',
        '2025_10_can_f1_r0_timing_racehistorychart_v01.pdf',
        '2025_10_can_f1_r0_timing_racelapchart_v01.pdf',
        2025,
        10,
        'race',
        '2025_10_race_classification.json',
        '2025_10_race_lap_times.json',
        nullcontext()
    ),
    (
        # Without some lap times PDF
        'https://www.fia.com/system/files/decision-document/2025_emilia_romagna_grand_prix_-_final_race_classification.pdf',
        '2025_07_ita_f1_r0_timing_racelapanalysis_v01_0.pdf',
        None,
        None,
        2025,
        7,
        'race',
        '2025_7_race_classification.json',
        None,
        pytest.raises(FileNotFoundError,
                      match='Lap chart, history chart, or lap time PDFs is missing')
    )
]


@pytest.fixture(params=race_list)
def prepare_race_data(request, tmp_path) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    # Download and parse race classification and lap times PDFs
    url_classification, url_lap_analysis, url_history_chart, url_lap_chart, year, round_no, \
        session, expected_classification, expected_lap_times, context = request.param
    pdfs = []
    if 'https://' not in url_classification:
        download_pdf('https://www.fia.com/sites/default/files/' + url_classification,
                     tmp_path / 'classification.pdf')
    else:
        download_pdf(url_classification, tmp_path / 'classification.pdf')
    pdfs.append(tmp_path / 'classification.pdf')
    if url_lap_analysis:
        download_pdf('https://www.fia.com/sites/default/files/' + url_lap_analysis,
                     tmp_path / 'lap_analysis.pdf')
        pdfs.append(tmp_path / 'lap_analysis.pdf')
    else:
        pdfs.append(None)
    if url_history_chart:
        download_pdf('https://www.fia.com/sites/default/files/' + url_history_chart,
                     tmp_path / 'history_chart.pdf')
        pdfs.append(tmp_path / 'history_chart.pdf')
    else:
        pdfs.append(None)
    if url_lap_chart:
        download_pdf('https://www.fia.com/sites/default/files/' + url_lap_chart,
                     tmp_path / 'lap_chart.pdf')
        pdfs.append(tmp_path / 'lap_chart.pdf')
    else:
        pdfs.append(None)
    with context:
        parser = RaceParser(*pdfs, year, round_no, session)
        classification_data = parser.classification_df.to_json()
        lap_times_data = None
        lap_times_data = parser.lap_times_df.to_json()
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

    # TODO: currently manually tested against fastf1 lap times. The test data should be generated
    #       automatically later. Also need to manually add if the lap time is deleted and if the
    #       lap is fastest manually. Also need to add the laps where PDFs have data but fastf1
    #       doesn't
    return classification_data, lap_times_data, expected_classification, expected_lap_times


def test_parse_race(prepare_race_data):
    classification_data, lap_times_data, expected_classification, expected_lap_times \
        = prepare_race_data
    assert classification_data == expected_classification

    # If no lap times (some lap times PDF is missing), skip the test for lap times
    if expected_lap_times is None:
        return

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

