# -*- coding: utf-8 -*-
"""Driver entry models"""
from pydantic import BaseModel, ConfigDict, PositiveInt

from models.foreign_key import Round


class Driver(BaseModel):
    car_number: PositiveInt
    name: str
    team: str

    model_config = ConfigDict(extra='forbid')


class RoundEntry(BaseModel):
    object_type: str = 'driver'
    foreign_keys: Round
    objects: list[Driver]

    model_config = ConfigDict(extra='forbid')
