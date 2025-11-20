# -*- coding: utf-8 -*-
"""
Test the parser against all PDFs in 2024. This assumes we have PDFs stored in `data/pdf` folder,
and have the naming convention of:

1. entry list: <year>_<round>_(revised or corrected_)entry_list.pdf
2. quali.: <year>_<round>_quali_<final or provisional>_classification.pdf and
           <year>_<round>_quali_lap_times.pdf
3. sprint quali.: <year>_<round>_sprint_quali_<final or provisional>_classification.pdf and
                  <year>_<round>_sprint_quali_lap_times.pdf
4. sprint race: <year>_<round>_sprint_<final or provisional>_classification.pdf,
                <year>_<round>_sprint_lap_times.pdf,
                <year>_<round>_sprint_lap_analysis.pdf, and
                <year>_<round>_sprint_lap_chart.pdf
5. race: <year>_<round>_race_<final or provisional>_classification.pdf,
         <year>_<round>_race_lap_times.pdf,
         <year>_<round>_race_lap_analysis.pdf, and
         <year>_<round>_race_lap_chart.pdf
"""
import pytest


def pytest_addoption(parser):
    parser.addoption(
        '--full', action='store_true',
        help='test if the parser works for all races in 2024. Only check if it runs without an '
             'error. The output correctness is NOT checked. Extremely slow'
    )  # TODO: provide details on how the PDFs should be downloaded


def pytest_configure(config):
    config.addinivalue_line('markers',
                            'full: test against all races in 2024. Extremely slow')


def pytest_runtest_setup(item):
    if 'full' in item.keywords and not item.config.getoption('--full'):
        pytest.skip('need --full option to run this test')
