# -*- coding: utf-8 -*-
"""Driver entry models"""
from pydantic import BaseModel, ConfigDict, PositiveInt

from .foreign_key import RoundEntry


class Driver(BaseModel):
    car_number: PositiveInt

    model_config = ConfigDict(extra='forbid')


class DriverData(BaseModel):
    object_type: str = 'RoundEntry'
    foreign_keys: RoundEntry

    objects: list[Driver]

    model_config = ConfigDict(extra='forbid')
