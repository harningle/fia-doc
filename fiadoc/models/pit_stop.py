# -*- coding: utf-8 -*-
"""
{
    "object": "pit_stop",
    "foreign_keys": {
        "season_year": 2023,
        "round_number": 22,
        "type": "R",
        "car_number": 1
    },
    "object": [
        {
            "lap": 16,
            "number": 1,
            "duration": timedelta(seconds=21, milliseconds=662),
            "local_timestamp": "17:27:36"
        },
        ...
    ]
}
"""
from datetime import timedelta

from pydantic import BaseModel, PositiveInt

from .foreign_key import SessionEntry


class PitStop(BaseModel):
    lap: PositiveInt     # Lap No.
    number: PositiveInt  # Car No.
    duration: dict[str, str | int]
    local_timestamp: str

    model_config = {'extra': 'forbid'}


class PitStopData(BaseModel):  # TODO: all xxxData can be combined into one class?
    object_type: str = 'pit_stop'
    foreign_keys: SessionEntry
    objects: list[PitStop]

    model_config = {'extra': 'forbid'}
