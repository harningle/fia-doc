"""Opening calendar time rows must not shift lap No."""
import pandas as pd
import pytest

from fiadoc.parser import QualifyingParser
from fiadoc.parser.page import ParsingError
from fiadoc.tests._warnings import assert_warnings

CLOCK_WARNING = 'Expected two opening calendar time rows'


def sector_rows(times, pits=(), car_no=43):
    return pd.DataFrame([
        {
            'car_no': car_no,
            'lap_no': lap,
            'lap_time': time,
            'pit': True if lap in pits else float('nan'),
            'lap_time_deleted': False,
            **{f'sector_{sector}_{field}': 30.0
               for sector in (1, 2, 3) for field in ('time', 'speed')},
        }
        for lap, time in enumerate(times, 1)
    ])


@pytest.mark.parametrize('times,pits,expected_numbers,warning', [
    (['15:00:00', '15:02:00', '1:30.000'], (), [1, 2], False),
    (['15:00:00', '1:30.000'], (1,), [0, 1], False),
    (['15:00:00', 'INCOMPLETE'], (), [1], True),
    (['INCOMPLETE'], (), [0], True),
    (['15:00:00', 'INCOMPLETE', '15:05:00', '1:30.000'], (), [1, 2, 3], False),
    (['15:00:06', '16:26:13', '16:28:04', '1:45.629'], (2,), [1, 2, 3], True),
])
def test_opening_clock_patterns(times, pits, expected_numbers, warning):
    with assert_warnings(required=[CLOCK_WARNING] if warning else []):
        result = QualifyingParser._sanitise_sector_analysis_df(sector_rows(times, pits))
    assert result.lap_no.tolist() == expected_numbers
    # Only the initial start marker is discarded, never the pit/out laps themselves
    assert result.pit.sum() == len(pits)
    assert result.iloc[-1].lap_time == times[-1]


@pytest.mark.parametrize('times,pits', [
    (['15:00:00', '15:02:00', '15:04:00', '1:30.000'], ()),
    (['15:00:00', '15:02:00', '15:04:00', '1:30.000'], (3,)),
    (['15:00:00', '15:02:00', '15:04:00', '15:06:00', '1:30.000'], (2,)),
    (['15:00:00', '1:30.000'], ()),
])
def test_unrecognised_clock_patterns_still_fail(times, pits):
    with pytest.warns(UserWarning, match=CLOCK_WARNING):
        with pytest.raises(ParsingError, match='Found drivers w/ first lap'):
            QualifyingParser._sanitise_sector_analysis_df(sector_rows(times, pits))


def test_clock_after_timed_lap_still_fails():
    with pytest.raises(AssertionError, match='calendar time as lap time in the middle'):
        QualifyingParser._sanitise_sector_analysis_df(
            sector_rows(['15:00:00', '15:02:00', '1:30.000', '15:05:00']))
