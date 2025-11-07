# -*- coding: utf-8 -*-
"""If we can handle usual drivers and create driver references for new guys not in Jolpica"""
import pytest

from fiadoc.drivers import Drivers

DRIVERS = Drivers()


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (2024, 'Max Verstappen', 'max_verstappen'),  # Normal
        (2025, 'LeWIs HamILtoN', 'hamilton'),        # Should be case agnostic
        (2023, 'Zhou Guanyu', 'zhou'),               # Edge case for Zhou
        (2024, 'GuaNYu zhOu', 'zhou')
    ]
)
def test_regular(year: int, full_name: str, expected: str):
    driver_id = DRIVERS.get(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (2025, 'Mick Schumacher', 'mick_schumacher'),
        (2025, 'Nyck de Vries', 'de_vries')
    ]
)
def test_not_in_current_year_but_exists_in_jolpica(year: int, full_name: str, expected: str):
    """E.g., Hulkenberg replaced Stroll in 2020 Eifel, but he was not a regular driver that year"""
    with pytest.warns(UserWarning,
                      match=f'Driver {full_name.lower()} not found in year {year} regular drivers '
                            f'mapping. Going to Jolpica for driver ID'):
        driver_id = DRIVERS.get(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (0, 'Lewis Hamilton', 'hamilton'),
        (1, 'Valtteri Bottas', 'bottas')
    ]
)
def test_not_in_maintained_years_but_exists_in_jolpica(year: int, full_name: str, expected: str):
    """In case we backfill earlier years' data and I don't want to manually maintain these years"""
    with pytest.warns(UserWarning,
                      match=f'Year {year} not maintained in regular drivers mapping. Going to '
                            f'Jolpica for driver ID'):
        driver_id = DRIVERS.get(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (2025, 'New Guy', 'new_guy'),
        (2024, 'Have Three Names', 'have_three_names')
    ]
)
def test_create_new_driver_id(year: int, full_name: str, expected: str):
    with pytest.warns(UserWarning,
                      match=f'Driver {full_name.lower()} not found in Jolpica. Creating a new '
                            f'driver ID "{expected}" for {full_name.lower()}'):
        driver_id = DRIVERS.get(year, full_name)
    assert driver_id == expected
