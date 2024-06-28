# -*- coding: utf-8 -*-
"""Frequently used foreign key models for the data objects"""
from typing import Literal

from pydantic import BaseModel, PositiveInt


class SessionEntry(BaseModel):
    year: PositiveInt
    round: PositiveInt
    type: str = Literal['R', 'Q', 'SR']  # Race, Quali, Sprint. TODO: enough?
    car_number: PositiveInt


class Round(BaseModel):
    year: PositiveInt
    round: PositiveInt
