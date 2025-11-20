# -*- coding: utf-8 -*-
"""If we can handle usual drivers and create driver references for new guys not in Jolpica"""
import json
from tempfile import TemporaryDirectory

import pytest
import requests_mock

from fiadoc.drivers import Drivers, BASE_URL

CACHE_DIR = TemporaryDirectory().name


@pytest.fixture
def drivers() -> Drivers:
    return Drivers(cache_dir=CACHE_DIR)


@pytest.fixture
def jolpica_mock(requests_mock: requests_mock.Mocker):
    """Mock Jolpica API endpoints"""
    with open('fiadoc/tests/fixtures/cached_drivers.json', 'r', encoding='utf8') as f:
        cached_drivers = json.loads(f.read())
    n_cached_drivers = str(len(cached_drivers))

    requests_mock.get(
        f'{BASE_URL}/drivers/?format=json&limit=1',
        json={'MRData': {'total': n_cached_drivers}}
    )
    requests_mock.get(
        f'{BASE_URL}/drivers',
        json={
            'MRData': {
                'total': n_cached_drivers,
                'DriverTable': {
                    'Drivers': cached_drivers
                }
            }
        }
    )


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (2024, 'Max Verstappen', 'max_verstappen'),  # Normal
        (2025, 'LeWIs HamILtoN', 'hamilton'),        # Should be case agnostic
        (2023, 'Zhou Guanyu', 'zhou'),               # Edge case for Zhou
        (2024, 'GuaNYu zhOu', 'zhou')
    ]
)
def test_regular(drivers: Drivers, year: int, full_name: str, expected: str):
    driver_id = drivers.get_driver_id(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (2025, 'Mick Schumacher', 'mick_schumacher'),
        (2025, 'Nyck de Vries', 'de_vries')
    ]
)
def test_not_in_current_year_but_exists_in_jolpica(
        drivers: Drivers, year: int, full_name: str, expected: str, jolpica_mock
):
    """E.g., Hulkenberg replaced Stroll in 2020 Eifel, but he was not a regular driver that year"""
    with pytest.warns(UserWarning,
                      match=f'Driver {full_name.lower()} not found in year {year} regular drivers '
                            f'mapping. Going to Jolpica for driver ID'):
        driver_id = drivers.get_driver_id(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (0, 'Lewis Hamilton', 'hamilton'),
        (1, 'Valtteri Bottas', 'bottas')
    ]
)
def test_not_in_maintained_years_but_exists_in_jolpica(
        drivers: Drivers, year: int, full_name: str, expected: str, jolpica_mock
):
    """In case we do a year whose regular drivers are not manually maintained"""
    with pytest.warns(UserWarning,
                      match=f'Year {year} not maintained in regular drivers mapping. Going to '
                            f'Jolpica for driver ID'):
        driver_id = drivers.get_driver_id(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (2025, 'New Guy', 'new_guy'),
        (2024, 'Have Three Names', 'have_three_names'),
        (2023, 'With Accént', 'with_accent'),
        (2025, "Has apostr‘oph’'e", 'has_apostr_oph_e')
    ]
)
def test_create_new_driver_id(drivers: Drivers, year: int, full_name: str, expected: str, jolpica_mock):
    with pytest.warns(Warning) as record:
        driver_id = drivers.get_driver_id(year, full_name)

    assert len(record) == 2
    assert record[0].message.args[0] == f'Driver {full_name.lower()} not found in year {year} ' \
                                        f'regular drivers mapping. Going to Jolpica for driver ID'
    assert record[1].message.args[0] == (f'Driver {full_name.lower()} not found in Jolpica. '
                                         f'Creating a new driver ID "{expected}" for '
                                         f'{full_name.lower()}')
    assert driver_id == expected


@pytest.mark.parametrize(
    'full_name, expected',
    [
        ('Jean-Éric Vergne', 'Jean-Éric'),
        ('Ha’rry James Potter', "Ha'rry"),
        ('Zhou Guanyu', 'Zhou')
    ]
)
def test_get_first_name(drivers: Drivers, full_name: str, expected: str):
    assert drivers.get_first_name(full_name) == expected


@pytest.mark.parametrize(
    'full_name, expected',
    [
        ('Jean-Éric Vergne', 'Vergne'),
        ('Ha’rry James Potter', 'James Potter'),
        ('Zhou Guanyu', 'Guanyu'),
        ('Patricio O’Ward', "O'Ward"),
        ("Patricio O'Ward", "O'Ward")
    ]
)
def test_get_last_name(drivers: Drivers, full_name: str, expected: str):
    assert drivers.get_last_name(full_name) == expected
