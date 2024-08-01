# -*- coding: utf-8 -*-
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, PositiveInt

from models.foreign_key import SessionEntry


class Lap(BaseModel):
    number: PositiveInt
    session: Literal[1, 2, 3]
    time: timedelta
    is_deleted: bool
    is_fastest_lap: bool


class LapData(BaseModel):
    object_type: str = 'lap'
    foreign_keys: SessionEntry
    objects: list[Lap]
