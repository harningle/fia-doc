import hashlib
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import pandas as pd
import pymupdf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

rc = {'figure.figsize': (8, 6),
      'axes.facecolor': 'white',  # Remove background colour
      'axes.grid': False,         # Turn on grid
      'axes.linewidth': '0.2',
      'axes.edgecolor': '0',      # Set axes edge color to be black
      'font.size': 2,
      'xtick.major.size': 1,
      'xtick.major.width': 0.2,
      'ytick.major.size': 1,
      'ytick.major.width': 0.2}
plt.rcdefaults()
plt.rcParams.update(rc)


def duration_to_millisecond(s: str | None) -> dict[str, str | int] | None:
    """Convert a time duration string to milliseconds

    >>> duration_to_millisecond('1:36:48.076')
    {'_type': 'timedelta', 'milliseconds': 5808076}
    >>> duration_to_millisecond('17:39.564')
    {'_type': 'timedelta', 'milliseconds': 1059564}
    >>> duration_to_millisecond('12.345')
    {'_type': 'timedelta', 'milliseconds': 12345}
    """
    if (s is None) or (s == '') or pd.isna(s):
        return None

    match s.count(':'):
        case 0:  # 12.345
            assert re.match(r'\d+\.\d+', s), f'{s} is not a valid time duration'
            sec, millisec = s.split('.')
            return {
                '_type': 'timedelta',
                'milliseconds': int(sec) * 1000 + int(millisec)
            }

        case 1:  # 1:23.456
            if m := re.match(r'(?P<minute>\d+):(?P<sec>\d+)\.(?P<millisec>\d+)', s):
                minute = int(m.group('minute'))
                sec = int(m.group('sec'))
                millisec = int(m.group('millisec'))
                return {
                    '_type': 'timedelta',
                    'milliseconds': minute * 60000 + sec * 1000 + millisec
                }
            else:
                raise ValueError(f'{s} is not a valid time duration')

        case 2:  # 1:23:45.678
            if m := re.match(r'(?P<hour>\d+):(?P<minute>\d+):(?P<sec>\d+)\.(?P<millisec>\d+)', s):
                hour = int(m.group('hour'))
                minute = int(m.group('minute'))
                sec = int(m.group('sec'))
                millisec = int(m.group('millisec'))
                return {
                    '_type': 'timedelta',
                    'milliseconds': hour * 3600000 + minute * 60000 + sec * 1000 + millisec
                }
            else:
                raise ValueError(f'{s} is not a valid time duration')

        case _:
            raise ValueError(f'{s} is not a valid time duration')


def time_to_timedelta(d: str) -> pd.Timedelta:
    """Parse a date string or a time duration string to pd.Timedelta

    TODO: not clear to me. May be confusing later. Either need better documentation, or make them
          into separate functions.

    There can be two possible input formats:

    1. hh:mm:ss, e.g. 18:05:42. This is simply the local calendar time
    2. mm:ss.SSS, e.g. 1:24.160. This is the lap time
    """
    n_colon = d.count(':')
    if n_colon == 2:  # noqa: PLR2004
        h, m, s = d.split(':')
        return pd.Timedelta(hours=int(h), minutes=int(m), seconds=int(s))
    elif n_colon == 1:
        m, s = d.split(':')
        s, ms = s.split('.')
        return pd.Timedelta(minutes=int(m), seconds=int(s), milliseconds=int(ms))
    else:
        raise ValueError(f'unknown date format: {d}')


def _default_cache_dir() -> Path:
    """
    Use `FIADOC_CACHE_DIR` env. var. if set. Otherwise, default to:

    * Windows: %LocalAppData%/fiadoc/Cache
    * macOS: ~/Library/Caches/fiadoc
    * Linux: ~/.cache/fiadoc
    """
    env = os.environ.get('FIADOC_CACHE_DIR')
    if env:
        cache_dir = Path(env)
    else:
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


def _default_download_cache_dir() -> Path:
    path = _default_cache_dir() / 'downloads'
    path.mkdir(parents=True, exist_ok=True)
    return path


def _pdf_cache_path(url: str) -> Path:
    return _default_download_cache_dir() / f'{hashlib.sha256(url.encode("utf-8")).hexdigest()}.pdf'


def _is_valid_pdf(content: bytes) -> bool:
    if b'%PDF-' not in content[:1024]:  # PyMuPDF can open other file types like HTML even with
        return False                    # `filetype='pdf'`... So check the bytes here
    try:
        pymupdf.open(stream=content, filetype='pdf')
        return True
    except:  # noqa: E722
        return False


def _download_session(n_retries: int = 3) -> requests.Session:
    retry = Retry(total=n_retries,
                  backoff_factor=1,
                  status_forcelist=(408, 425, 429, 500, 502, 503, 504),
                  allowed_methods=('GET',),
                  raise_on_status=False)
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session


def download_pdf(url: str, out_path: str | os.PathLike, n_retries: int = 3) -> None:
    """
    Download a PDF file from the given URL with caching and validation.

    PDFs are cached locally based on URL hash. Subsequent downloads of the same URL will use the
    cached version if it exists.
    """
    out_path = Path(out_path)
    cache_path = _pdf_cache_path(url)
    if cache_path.exists():
        shutil.copy2(cache_path, out_path)
        return

    last_error: Optional[Exception] = None
    session = _download_session(n_retries=n_retries)
    for attempt in range(n_retries):
        try:
            resp = session.get(url)
            resp.raise_for_status()
            if not _is_valid_pdf(resp.content):
                last_error = pymupdf.FileDataError(f'Failed to open the PDF at {url}')
            else:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, 'wb') as f:
                    f.write(resp.content)
                shutil.copy2(cache_path, out_path)
                return
        except requests.RequestException as e:
            last_error = e
        time.sleep(attempt + 1)
    raise last_error


def sort_json(j: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort a list of dicts by their keys and values, for comparison used in tests"""
    def _sort_recursively(i: Any) -> Any:
        """Helper function to sort lists/dicts recursively"""
        if isinstance(i, dict):
            return {k: _sort_recursively(v) for k, v in sorted(i.items())}
        elif isinstance(i, list):
            return sorted([_sort_recursively(k) for k in i], key=str)
        else:
            return i
    return _sort_recursively(j)


def _pd_concat(objs: list[Optional[pd.DataFrame]], ignore_index: bool = True) -> pd.DataFrame:
    """
    To silence "FutureWarning: The behavior of DataFrame concatenation with empty or all-NA
    entries is deprecated"
    """
    cleaned_objs = []
    for obj in objs:
        if (obj is not None) and (not obj.empty):
            obj = obj.dropna(axis=1, how='all')  # Drop all empty cols.  # noqa: PLW2901
            if not obj.empty:
                cleaned_objs.append(obj)
    if not cleaned_objs:
        raise ValueError('All dfs. are empty or None')
    return pd.concat(cleaned_objs, ignore_index=ignore_index)
