# -*- coding: utf-8 -*-
"""
{
    "object": "lap",
    "foreign_keys": {
        "season_year": 2023,
        "round_number": 22,
        "type": "R",
        "car_number": 1
    },
    "object": [
        {
            "number": 1,
            "position": 1
            "time": timedelta(minutes=1, seconds=32, milliseconds=190)
        },
        {
            "number": 2,
            "position": 1
            "time": timedelta(minutes=1, seconds=30, milliseconds=710)
        },
        ...
    ]
}
"""
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, PositiveInt

from models.foreign_key import SessionEntry


class Lap(BaseModel):
    number: PositiveInt
    position: PositiveInt
    time: timedelta


class LapData(BaseModel):
    object_type: str = 'lap'
    foreign_keys: SessionEntry
    objects: list[Lap]
