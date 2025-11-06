# -*- coding: utf-8 -*-
"""Class dealing with driver full name to ID mapping"""
import json
import os
import sys
import time
import warnings
from functools import cached_property
from pathlib import Path
from typing import Optional

import requests

BASE_URL = 'https://api.jolpi.ca/ergast/f1'


class Drivers:
    def __init__(self, cache_dir: Optional[str | os.PathLike] = None):
        if cache_dir:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = self._default_cache_dir

    @cached_property
    def regular_drivers(self) -> dict[int, dict[str, str]]:
        """Manually maintained once a year for regular drivers. Will default to this to speed up"""
        return {
            2023: {
                'Max Verstappen': 'max_verstappen',
                'Sergio Perez': 'perez',
                'Charles Leclerc': 'leclerc',
                'Carlos Sainz': 'sainz',
                'George Russell': 'russell',
                'Lewis Hamilton': 'hamilton',
                'Esteban Ocon': 'ocon',
                'Pierre Gasly': 'gasly',
                'Oscar Piastri': 'piastri',
                'Lando Norris': 'norris',
                'Valtteri Bottas': 'bottas',
                'Zhou Guanyu': 'zhou',
                'Lance Stroll': 'stroll',
                'Fernando Alonso': 'alonso',
                'Kevin Magnussen': 'kevin_magnussen',
                'Nico Hulkenberg': 'hulkenberg',
                'Daniel Ricciardo': 'ricciardo',
                'Yuki Tsunoda': 'tsunoda',
                'Alexander Albon': 'albon',
                'Logan Sargeant': 'sargeant',
                'Nyck de Vries': 'de_vries',
                'Liam Lawson': 'lawson'
            },
            2024: {
                'Max Verstappen': 'max_verstappen',
                'Sergio Perez': 'perez',
                'George Russell': 'russell',
                'Lewis Hamilton': 'hamilton',
                'Charles Leclerc': 'leclerc',
                'Carlos Sainz': 'sainz',
                'Oscar Piastri': 'piastri',
                'Lando Norris': 'norris',
                'Lance Stroll': 'stroll',
                'Fernando Alonso': 'alonso',
                'Esteban Ocon': 'ocon',
                'Pierre Gasly': 'gasly',
                'Alexander Albon': 'albon',
                'Logan Sargeant': 'sargeant',
                'Daniel Ricciardo': 'ricciardo',
                'Yuki Tsunoda': 'tsunoda',
                'Valtteri Bottas': 'bottas',
                'Zhou Guanyu': 'zhou',
                'Kevin Magnussen': 'kevin_magnussen',
                'Nico Hulkenberg': 'hulkenberg',
                'Oliver Bearman': 'bearman',
                'Franco Colapinto': 'colapinto',
                'Jack Doohan': 'doohan',
                'Liam Lawson': 'lawson',
                'Andrea Kimi Antonelli': 'antonelli'
            },
            2025: {
                'Oscar Piastri': 'piastri',
                'Lando Norris': 'norris',
                'Charles Leclerc': 'leclerc',
                'Lewis Hamilton': 'hamilton',
                'Max Verstappen': 'max_verstappen',
                'Liam Lawson': 'lawson',
                'George Russell': 'russell',
                'Andrea Kimi Antonelli': 'antonelli',
                'Kimi Antonelli': 'antonelli',
                'Lance Stroll': 'stroll',
                'Fernando Alonso': 'alonso',
                'Pierre Gasly': 'gasly',
                'Jack Doohan': 'doohan',
                'Esteban Ocon': 'ocon',
                'Oliver Bearman': 'bearman',
                'Isack Hadjar': 'hadjar',
                'Yuki Tsunoda': 'tsunoda',
                'Alexander Albon': 'albon',
                'Carlos Sainz': 'sainz',
                'Nico Hulkenberg': 'hulkenberg',
                'Gabriel Bortoleto': 'bortoleto',
                'Garbiel Bortoleto': 'bortoleto',  # Typo in entry list in 2025 Australian
                'Franco Colapinto': 'colapinto'
            }
        }

    @cached_property
    def _default_cache_dir(self) -> Path:
        match sys.platform:
            case 'linux':
                cache_dir = Path.home() / '.cache' / 'fiadoc'
            case 'darwin':
                cache_dir = Path.home() / 'Library' / 'Caches' / 'fiadoc'
            case 'win32':
                cache_dir = Path.home() / 'AppData' / 'Local' / 'fiadoc' / 'Cache'
            case _:
                raise NotImplementedError(f'Unsupported platform: {sys.platform}')
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir

    @property
    def cached_drivers(self) -> dict[str, str]:
        cached_file = self._cache_dir / 'driver_mapping.json'
        if cached_file.exists():
            self._refresh_cache()
        else:
            drivers = self._get_all_drivers_from_jolpica()
            with open(cached_file, 'w', encoding='utf8') as f:
                json.dump(drivers, f, indent=4)

        with open(cached_file, 'r', encoding='utf8') as f:
            return json.load(f)

    @classmethod
    def _get_driver_count(cls):
        """
        Get the total #. of drivers from jolpica API. Will use this count as the sole criteria to
        decide whether to refresh the local cache
        """
        url = f'{BASE_URL}/drivers/?format=json&limit=1'
        resp = requests.get(url)
        if resp.status_code != 200:
            raise JolpicaApiError(f'/ergast/f1/drivers failure: {resp.status_code}\n{resp.text}')
        return int(resp.json()['MRData']['total'])

    @classmethod
    def _get_all_drivers_from_jolpica(cls) -> dict[str, str]:
        """
        Get all drivers from Jolpica and return as a driver full name to ID dict.
        """
        drivers = {}
        limit = 100
        offset = 0
        params = {'format': 'json', 'limit': limit, 'offset': offset}

        while True:
            resp = requests.get(f'{BASE_URL}/drivers', params=params)
            if resp.status_code != 200:
                raise JolpicaApiError(f'/ergast/f1/drivers failure: '
                                      f'{resp.status_code}\n{resp.text}')
            data = resp.json()

            driver_list = data['MRData']['DriverTable']['Drivers']
            if not driver_list:
                break
            for driver in driver_list:
                if driver['driverId'].lower() == 'zhou':  # Exception for ZHOU Guanyu
                    full_name = 'Zhou Guanyu'
                else:
                    full_name = f'{driver['givenName']} {driver['familyName']}'
                drivers[full_name] = driver['driverId']

            offset += limit
            if offset >= int(data['MRData']['total']):
                break
            time.sleep(0.25)  # Rate limit 4 requests per second
        return drivers

    def _refresh_cache(self) -> None:
        """Refresh cached driver mapping ONLY IF Jolpica's total driver count is different"""
        cached_drivers = self.cached_drivers
        jolpica_total_count = self._get_driver_count()
        if len(cached_drivers) == jolpica_total_count:
            return

        drivers = self._get_all_drivers_from_jolpica()
        cached_file = self._cache_dir / 'driver_mapping.json'
        with open(cached_file, 'w', encoding='utf8') as f:
            json.dump(drivers, f, indent=4)
        return

    def get(self, year: int, full_name: str) -> str:
        """Get driver ID from full name"""
        if year in self.regular_drivers and full_name in self.regular_drivers[year]:
            return self.regular_drivers[year][full_name]
        elif year not in self.regular_drivers:
            warnings.warn(f'Year {year} not maintained in regular drivers mapping. Going to '
                          f'Jolpica for driver ID')
        else:
            warnings.warn(f'Driver {full_name} not found in year {year} regular drivers mapping. '
                          f'Going to Jolpica for driver ID')

        drivers = self.cached_drivers
        if full_name in drivers:
            return drivers[full_name]
        else:
            driver_id = self.create_driver_id(full_name)
            warnings.warn(f'Driver {full_name} not found in Jolpica. Creating a new driver ID '
                          f'"{driver_id}" for {full_name}')
            return driver_id

    @staticmethod
    def create_driver_id(full_name: str) -> str:
        """Create an ID for new drivers that are not in Jolpica DB. Format is first_last"""
        return '_'.join(full_name.lower().split(' '))


class JolpicaApiError(Exception):
    pass


if __name__ == '__main__':
    pass
