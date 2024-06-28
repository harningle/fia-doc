# -*- coding: utf-8 -*-
from datetime import timedelta

from pydantic import BaseModel, PositiveInt

from models.foreign_key import SessionEntry


class Lap(BaseModel):
    lap_number: PositiveInt
    position: PositiveInt
    time: timedelta


class FastestLap(BaseModel):
    lap_number: PositiveInt
    fastest_lap_rank: PositiveInt


class LapData(BaseModel):
    object_type: str = 'lap'
    foreign_keys: SessionEntry
    objects: list[Lap]


class FastestLapData(BaseModel):
    object_type: str = 'fastest_lap'
    foreign_keys: SessionEntry
    objects: FastestLap
