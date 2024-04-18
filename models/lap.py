# -*- coding: utf-8 -*-
"""
{
    "object": "lap",
    "foreign_keys": {
        "round_number": 22,
        "season_year": 2023,
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

from pydantic import BaseModel


class Lap(BaseModel):
    number: int
    position: int
    # time: timedelta


class SessionEntry(BaseModel):
    round_number: int
    season_year: int
    team: str
    driver: str


class LapData(BaseModel):
    object: str = 'lap'
    foreign_keys: SessionEntry
    data: list[Lap]
