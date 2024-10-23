import re

import pandas as pd


def duration_to_millisecond(s: str) -> int:
    """Convert a time duration string to milliseconds

    >>> duration_to_millisecond('1:36:48.076')
    5808076
    >>> duration_to_millisecond('17:39.564')
    1059564
    """
    pat = re.compile(r'(?:(?P<hour>\d+):)?(?P<minute>\d{2}):(?P<sec>\d{2})\.(?P<millisec>\d{3})')
    if m := pat.match(s):
        hour = int(m.group('hour') or 0)
        minute = int(m.group('minute'))
        second = int(m.group('sec'))
        millisecond = int(m.group('millisec'))
        return hour * 3600000 + minute * 60000 + second * 1000 + millisecond
    else:
        raise ValueError(f'{s} is not a valid time duration')


