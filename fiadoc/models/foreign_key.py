# -*- coding: utf-8 -*-
"""Frequently used foreign key models for the data objects"""
from typing import Literal
from typing_extensions import Self

from pydantic import BaseModel, ConfigDict, PositiveInt, field_validator, model_validator

from .._constants import DRIVERS, TEAMS


class SessionEntryForeignKeys(BaseModel):
    year: PositiveInt
    round: PositiveInt
    session: str = Literal['R', 'Q1', 'Q2', 'Q3', 'SR', 'SQ1', 'SQ2', 'SQ3', 'FP1', 'FP2', 'FP3']
    car_number: PositiveInt

    @field_validator('session')
    @classmethod
    def clean_session(cls, session: str) -> str:
        match session.lower().strip():
            case 'r' | 'q1' | 'q2' | 'q3' | 'sr' | 'sq1' | 'sq2' | 'sq3' | 'fp1' | 'fp2' | 'fp3':
                return session.upper()
            case 'race':  # Some simple mapping
                return 'R'
            case 'sprint' | 'sprint_race' | 'sprint race':
                return 'SR'
            case _:
                raise ValueError(f'Invalid session: {session}. Must be one of: "R", "Q1", "Q2",'
                                 f'"Q3", "SR", "SQ1", "SQ2", "SQ3", "FP1", "FP2", "FP3"')

    model_config = ConfigDict(extra='forbid')


class RoundEntry(BaseModel):
    year: PositiveInt
    round: PositiveInt
    driver_reference: str
    team_reference: str

    @model_validator(mode='before')
    def get_team_reference(self) -> Self:
        if self['year'] in TEAMS:
            if self['team_reference'] in TEAMS[self['year']]:
                self['team_reference'] = TEAMS[self['year']][self['team_reference']]
                return self
            else:
                raise ValueError(f"team {self['team_reference']} not found in year "
                                 f"{self['year']}'s team name mapping. Available teams: "
                                 f"{TEAMS[self['year']].keys()}")
        else:
            raise ValueError(f"year {self['year']} not found in team name mapping. Available "
                             f'years: {TEAMS.keys()}')

    @model_validator(mode='before')
    def get_driver_name(self) -> Self:
        if self['year'] in DRIVERS:
            if self['driver_reference'] in DRIVERS[self['year']]:
                self['driver_reference'] = DRIVERS[self['year']][self['driver_reference']]
                return self
            else:
                raise ValueError(f"driver {self['driver_reference']} not found in year "
                                 f"{self['year']}'s driver name mapping. Available drivers: "
                                 f"{DRIVERS[self['year']].keys()}")
        else:
            raise ValueError(f"year {self['year']} not found in driver name mapping. Available "
                             f'years: {DRIVERS.keys()}')

    model_config = ConfigDict(extra='forbid')


class PitStopEntry(BaseModel):
    year: PositiveInt
    round: PositiveInt
    session: str = Literal['R', 'SR', 'FP1', 'FP2', 'FP3']
    car_number: PositiveInt
    lap: PositiveInt

    @field_validator('session')
    @classmethod
    def clean_session(cls, session: str) -> str:
        match session.lower().strip():
            case 'r' | 'sr' | 'fp1' | 'fp2' | 'fp3':
                return session.upper()
            case 'race':  # Some simple mapping
                return 'R'
            case 'sprint' | 'sprint_race' | 'sprint race':
                return 'SR'
            case _:
                raise ValueError(f'Invalid session: {session}. Must be one of: "R", "SR", "FP1", '
                                 f'"FP2", "FP3"')

    model_config = ConfigDict(extra='forbid')
