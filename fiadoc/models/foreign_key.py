# -*- coding: utf-8 -*-
"""Frequently used foreign key models for the data objects"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator


class SessionEntry(BaseModel):
    year: PositiveInt
    round: PositiveInt
    session: str = Literal['R', 'Q1', 'Q2', 'Q3', 'SR', 'SQ1', 'SQ2', 'SQ3', 'FP1', 'FP2', 'FP3']
    car_number: PositiveInt

    @field_validator('session')
    @classmethod
    def clean_session(cls, session: str) -> str:
        match session:
            case 'R' | 'Q1' | 'Q2' | 'Q3' | 'SR' | 'SQ1' | 'SQ2' | 'SQ3' | 'FP1' | 'FP2' | 'FP3':
                return session
            case 'race':
                return 'R'
            case 'sprint':
                return 'SR'
            case _:
                raise ValueError(f'Invalid session: {session}. Must be one of: "R", "Q1", "Q2",'
                                 f'"Q3", "SR", "SQ1", "SQ2", "SQ3", "FP1", "FP2", "FP3"')

    model_config = ConfigDict(extra='forbid')


class Round(BaseModel):
    year: PositiveInt
    round: PositiveInt

    model_config = ConfigDict(extra='forbid')