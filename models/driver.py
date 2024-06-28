# -*- coding: utf-8 -*-
"""Driver entry models"""
from typing import Literal

from pydantic import BaseModel, PositiveInt

from models.foreign_key import Round


class Driver(BaseModel):
    car_number: PositiveInt
    name: str
    team: str
    role: str = Literal['permanent', 'reserve']


class RoundEntry(BaseModel):
    object_type: str = 'driver'
    foreign_keys: Round
    objects: list[Driver]
