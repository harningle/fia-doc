# -*- coding: utf-8 -*-
"""If we can handle usual drivers and create driver references for new guys not in Jolpica"""
import pytest

from fiadoc.drivers import Drivers


@pytest.fixture
def drivers() -> Drivers:
    return Drivers()

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
    driver_id = drivers.get(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (2025, 'Mick Schumacher', 'mick_schumacher'),
        (2025, 'Nyck de Vries', 'de_vries')
    ]
)
def test_not_in_current_year_but_exists_in_jolpica(
        drivers: Drivers, year: int, full_name: str, expected: str
):
    """E.g., Hulkenberg replaced Stroll in 2020 Eifel, but he was not a regular driver that year"""
    with pytest.warns(UserWarning,
                      match=f'Driver {full_name.lower()} not found in year {year} regular drivers '
                            f'mapping. Going to Jolpica for driver ID'):
        driver_id = drivers.get(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (0, 'Lewis Hamilton', 'hamilton'),
        (1, 'Valtteri Bottas', 'bottas')
    ]
)
def test_not_in_maintained_years_but_exists_in_jolpica(
        drivers: Drivers, year: int, full_name: str, expected: str
):
    """In case we do a year whose regular drivers are not manually maintained"""
    with pytest.warns(UserWarning,
                      match=f'Year {year} not maintained in regular drivers mapping. Going to '
                            f'Jolpica for driver ID'):
        driver_id = drivers.get(year, full_name)
    assert driver_id == expected


@pytest.mark.parametrize(
    'year, full_name, expected',
    [
        (2025, 'New Guy', 'new_guy'),
        (2024, 'Have Three Names', 'have_three_names')
    ]
)
def test_create_new_driver_id(drivers: Drivers, year: int, full_name: str, expected: str):
    with pytest.warns(Warning) as record:
        driver_id = drivers.get(year, full_name)

    assert len(record) == 2
    assert record[0].message.args[0] == f'Driver {full_name.lower()} not found in year {year} ' \
                                        f'regular drivers mapping. Going to Jolpica for driver ID'
    assert record[1].message.args[0] == (f'Driver {full_name.lower()} not found in Jolpica. '
                                         f'Creating a new driver ID "{expected}" for '
                                         f'{full_name.lower()}')
    assert driver_id == expected
