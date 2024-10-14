import re

import pandas as pd


def parse_duration(s: str) -> pd.Timedelta:
    """Convert a time duration string to `pd.Timedelta` object

    >>> parse_duration('1:36:48.076')
    Timedelta('0 days 01:36:48.076000')
    >>> parse_duration('17:39.564')
    Timedelta('0 days 00:17:39.564000')
    """
    has_hour_pat = re.compile(r'^\d{1,2}:\d{2}:\d{2}\.\d{3}$')
    no_hour_pat = re.compile(r'^\d{2}:\d{2}\.\d{3}$')
    if has_hour_pat.match(s):
        return pd.Timedelta(s)
    elif no_hour_pat.match(s):
        return pd.Timedelta('0:' + s)
    else:
        raise ValueError(f'{s} is not a valid time duration')
