import os
import re
import tempfile
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from string import printable
from types import SimpleNamespace
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pymupdf
import requests

# from PIL import Image
from .models.foreign_key import SessionEntryForeignKeys
from .models.lap import LapImport, LapObject


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
    if s is None:
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


def download_pdf(url: str, out_path: str | os.PathLike) -> None:
    """
    Download a PDF file from the given URL. This downloads PDFs for testing. This is a temporary
    solution. Will be deleted when Philipp's package is ready
    """
    resp = requests.get(url)
    with open(out_path, 'wb') as f:
        f.write(resp.content)
    return


def quali_lap_times_to_json(df, year, round_no, session) -> list[dict]:
    lap_data = []
    # TODO: first lap's lap time is calendar time, not lap time, so drop it
    # Lap No. can be missing (e.g.#47)
    df = df[(df.lap_no >= 2) | df.lap_no.isna()].copy()  # noqa: PLR2004
    df.lap_time = df.lap_time.apply(duration_to_millisecond)
    for q in [1, 2, 3]:
        temp = df[df.Q == q].copy()
        temp['lap'] = temp.apply(
            lambda x: LapObject(
                number=x.lap_no,
                time=x.lap_time,
                is_deleted=x.lap_time_deleted,
                is_entry_fastest_lap=x.is_fastest_lap
            ),
            axis=1
        )
        temp = temp.groupby('car_no')[['lap']].agg(list).reset_index()
        temp['session_entry'] = temp['car_no'].map(
            lambda x: SessionEntryForeignKeys(
                year=year,
                round=round_no,
                session=f'Q{q}' if session == 'quali' else f'SQ{q}',
                car_number=x
            )
        )
        temp['lap_data'] = temp.apply(
            lambda x: LapImport(
                object_type="Lap",
                foreign_keys=x['session_entry'],
                objects=x['lap']
            ).model_dump(exclude_unset=True),
            axis=1
        )
        lap_data.extend(temp['lap_data'].tolist())
    return lap_data
