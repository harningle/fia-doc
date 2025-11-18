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
from filelock import FileLock

BASE_URL = 'https://api.jolpi.ca/ergast/f1'
TIMEOUT = 10  # Seconds


class Drivers:
    """
    Class to handle driver full name to ID mapping. Will first look up the manually maintained
    regular drivers. If not found, will check (locally cached and the latest) Jolpica. If still not
    found, create a new driver ID based on the full name.
    """
    def __init__(self, cache_dir: Optional[str | os.PathLike] = None):
        """Initialise Drivers class

        :param cache_dir: Path to cache directory. The default is
                          * Windows: %LocalAppData%/fiadoc/Cache
                          * macOS: ~/Library/Caches/fiadoc
                          * Linux: ~/.cache/fiadoc
        """
        if cache_dir:
            self._cache_dir = Path(cache_dir)
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self._cache_dir = self._default_cache_dir

    @cached_property
    def regular_drivers(self) -> dict[int, dict[str, str]]:
        """Manually maintained once a year for regular drivers. Will default to this to speed up"""
        # TODO: move this huge dict to a json file?
        return {
            2023: {
                'max verstappen': 'max_verstappen',
                'sergio perez': 'perez',
                'charles leclerc': 'leclerc',
                'carlos sainz': 'sainz',
                'george russell': 'russell',
                'lewis hamilton': 'hamilton',
                'esteban ocon': 'ocon',
                'pierre gasly': 'gasly',
                'oscar piastri': 'piastri',
                'lando norris': 'norris',
                'valtteri bottas': 'bottas',
                'zhou guanyu': 'zhou',
                'lance stroll': 'stroll',
                'fernando alonso': 'alonso',
                'kevin magnussen': 'kevin_magnussen',
                'nico hulkenberg': 'hulkenberg',
                'daniel ricciardo': 'ricciardo',
                'yuki tsunoda': 'tsunoda',
                'alexander albon': 'albon',
                'logan sargeant': 'sargeant',
                'nyck de vries': 'de_vries',
                'liam lawson': 'lawson'
            },
            2024: {
                'max verstappen': 'max_verstappen',
                'sergio perez': 'perez',
                'george russell': 'russell',
                'lewis hamilton': 'hamilton',
                'charles leclerc': 'leclerc',
                'carlos sainz': 'sainz',
                'oscar piastri': 'piastri',
                'lando norris': 'norris',
                'lance stroll': 'stroll',
                'fernando alonso': 'alonso',
                'esteban ocon': 'ocon',
                'pierre gasly': 'gasly',
                'alexander albon': 'albon',
                'logan sargeant': 'sargeant',
                'daniel ricciardo': 'ricciardo',
                'yuki tsunoda': 'tsunoda',
                'valtteri bottas': 'bottas',
                'zhou guanyu': 'zhou',
                'kevin magnussen': 'kevin_magnussen',
                'nico hulkenberg': 'hulkenberg',
                'oliver bearman': 'bearman',
                'franco colapinto': 'colapinto',
                'jack doohan': 'doohan',
                'liam lawson': 'lawson',
                'andrea kimi antonelli': 'antonelli'
            },
            2025: {
                'oscar piastri': 'piastri',
                'lando norris': 'norris',
                'charles leclerc': 'leclerc',
                'lewis hamilton': 'hamilton',
                'max verstappen': 'max_verstappen',
                'liam lawson': 'lawson',
                'george russell': 'russell',
                'andrea kimi antonelli': 'antonelli',
                'kimi antonelli': 'antonelli',
                'lance stroll': 'stroll',
                'fernando alonso': 'alonso',
                'pierre gasly': 'gasly',
                'jack doohan': 'doohan',
                'esteban ocon': 'ocon',
                'oliver bearman': 'bearman',
                'isack hadjar': 'hadjar',
                'yuki tsunoda': 'tsunoda',
                'alexander albon': 'albon',
                'carlos sainz': 'sainz',
                'nico hulkenberg': 'hulkenberg',
                'gabriel bortoleto': 'bortoleto',
                'garbiel bortoleto': 'bortoleto',  # typo in entry list in 2025 australian
                'franco colapinto': 'colapinto'
            }
        }

    @cached_property
    def _default_cache_dir(self) -> Path:
        """
        * Windows: %LocalAppData%/fiadoc/Cache
        * macOS: ~/Library/Caches/fiadoc
        * Linux: ~/.cache/fiadoc
        """
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

    @cached_property
    def cached_drivers(self) -> dict[str, str]:
        """Get cached drivers from Jolpica, and refresh it if needed"""
        cached_file = self._cache_dir / 'driver_mapping.json'
        lock_file = self._cache_dir / 'driver_mapping.json.lock'
        with FileLock(lock_file, timeout=60):
            if cached_file.exists():
                self._refresh_cache()
                with open(cached_file, 'r', encoding='utf8') as f:
                    return json.load(f)
            else:
                drivers = self._get_all_drivers_from_jolpica()
                with open(cached_file.with_suffix('.tmp'), 'w', encoding='utf8') as f:
                    json.dump(drivers, f, indent=4)
                os.replace(cached_file.with_suffix('.tmp'), cached_file)  # Atomic write
                return drivers

    @staticmethod
    def _get_driver_count():
        """
        Get the total #. of drivers from jolpica API. Will use this count as the sole criteria to
        decide whether to refresh the local cache
        """
        url = f'{BASE_URL}/drivers/?format=json&limit=1'
        try:
            resp = requests.get(url, timeout=TIMEOUT)
        except requests.exceptions.RequestException as e:
            raise JolpicaApiError(f'/ergast/f1/drivers failure: {e}')
        time.sleep(0.25)  # Rate limit
        return int(resp.json()['MRData']['total'])

    @staticmethod
    def _get_all_drivers_from_jolpica() -> dict[str, str]:
        """
        Get all drivers from Jolpica and return as a driver full name to ID dict.
        """
        drivers = {}
        limit = 100
        offset = 0
        params = {'format': 'json', 'limit': limit, 'offset': offset}

        while True:
            params['offset'] = offset
            try:
                resp = requests.get(f'{BASE_URL}/drivers', params=params, timeout=TIMEOUT)
            except requests.exceptions.RequestException as e:
                raise JolpicaApiError(f'/ergast/f1/drivers failure: {e}')
            data = resp.json()

            driver_list = data['MRData']['DriverTable']['Drivers']
            if not driver_list:
                break
            for driver in driver_list:
                if driver['driverId'].lower() == 'zhou':  # Exception for ZHOU Guanyu
                    full_name = 'Zhou Guanyu'
                else:
                    full_name = f'{driver['givenName']} {driver['familyName']}'
                drivers[full_name.lower()] = driver['driverId']

            offset += limit
            if offset >= int(data['MRData']['total']):
                break
            time.sleep(0.25)  # Rate limit 4 requests per second
        return drivers

    def _refresh_cache(self, force_overwrite: bool = False) -> None:
        """Refresh cached driver mapping ONLY IF Jolpica's total driver count is different"""
        cached_file = self._cache_dir / 'driver_mapping.json'

        if not force_overwrite:
            try:
                with open(cached_file, 'r', encoding='utf8') as f:
                    cached_drivers = json.load(f)
                jolpica_total_count = self._get_driver_count()
                if len(cached_drivers) == jolpica_total_count:
                    return
            except Exception as e:
                warnings.warn(f'Failed to read existing driver cache; force refreshing it: {e}')

        # If reaches here, either we want to force overwrite/update the cache, or something is
        # wrong with the existing cache (e.g., JSONDecodeError)
        drivers = self._get_all_drivers_from_jolpica()
        with open(cached_file, 'w', encoding='utf8') as f:
            json.dump(drivers, f, indent=4)
        return

    def get(self, year: int, full_name: str) -> str:
        """Get driver ID from full name"""
        # Exception for ZHOU Guanyu
        full_name = full_name.lower()
        if full_name == 'guanyu zhou':
            full_name = 'zhou guanyu'

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
