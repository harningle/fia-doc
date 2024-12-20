# -*- coding: utf-8 -*-
from pydantic import BaseModel, ConfigDict, PositiveInt

from .foreign_key import SessionEntry


class Lap(BaseModel):
    number: PositiveInt
    position: PositiveInt
    time: dict[str, str | int]
    is_entry_fastest_lap: bool

    model_config = ConfigDict(extra='forbid')


class QualiLap(BaseModel):
    number: PositiveInt
    time: dict[str, str | int]
    is_deleted: bool
    is_entry_fastest_lap: bool

    model_config = ConfigDict(extra='forbid')


class LapData(BaseModel):
    object_type: str = 'lap'
    foreign_keys: SessionEntry
    objects: list[Lap | QualiLap]

    model_config = ConfigDict(extra='forbid')
