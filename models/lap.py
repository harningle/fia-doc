# -*- coding: utf-8 -*-
"""
{
    "object": "lap",
    "foreign_keys": {
        "season_year": 2023,
        "round_number": 22,
        "type": "R",
        "team": "Red Bull",
        "driver": "Max Verstappen"
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

from pydantic import BaseModel


class Lap(BaseModel):
    number: int
    position: int
    # time: timedelta


class SessionEntry(BaseModel):
    season_year: int
    round_number: int
    type: str = Literal['R', 'Q', 'SR']  # Race, Quali, Sprint. TODO: enough?
    team: str
    driver: str


class LapData(BaseModel):
    object_type: str = 'lap'
    foreign_keys: SessionEntry
    objects: list[Lap]
