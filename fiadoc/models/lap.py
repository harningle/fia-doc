# -*- coding: utf-8 -*-
from datetime import timedelta

from pydantic import BaseModel, ConfigDict, PositiveInt

from .foreign_key import SessionEntry


class Lap(BaseModel):
    number: PositiveInt
    position: PositiveInt
    time: timedelta

    model_config = ConfigDict(extra='forbid')


class QualiLap(BaseModel):
    number: PositiveInt
    time: timedelta
    is_deleted: bool
    is_entry_fastest_lap: bool

    model_config = ConfigDict(extra='forbid')


class FastestLap(BaseModel):
    number: PositiveInt
    fastest_lap_rank: PositiveInt

    model_config = ConfigDict(extra='forbid')


class LapData(BaseModel):
    object_type: str = 'lap'
    foreign_keys: SessionEntry
    objects: list[Lap | QualiLap]

    model_config = ConfigDict(extra='forbid')


class FastestLapData(BaseModel):
    object_type: str = 'lap'
    foreign_keys: SessionEntry
    objects: list[FastestLap]

    model_config = ConfigDict(extra='forbid')
