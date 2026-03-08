# -*- coding: utf-8 -*-
"""Frequently used foreign key models for the data objects"""
from jolpica.schemas import data_import
from pydantic import ConfigDict, field_validator, model_validator

from .._constants import TEAMS


class SessionValidatorMixin:
    @field_validator('session')
    @classmethod
    def clean_session(cls, session: str) -> str:
        match session.lower().strip():
            case (
                "r"
                | "q1"
                | "q2"
                | "q3"
                | "sr"
                | "sq1"
                | "sq2"
                | "sq3"
                | "fp1"
                | "fp2"
                | "fp3"
            ):
                return session.upper()
            case "race":  # Some simple mapping
                return "R"
            case "sprint" | "sprint_race" | "sprint race":
                return "SR"
            case _:
                raise ValueError(
                    f'Invalid session: {session}. Must be one of: "R", "Q1", "Q2",'
                    f'"Q3", "SR", "SQ1", "SQ2", "SQ3", "FP1", "FP2", "FP3"'
                )


class SessionEntryForeignKeys(
    data_import.SessionEntryForeignKeys, SessionValidatorMixin
):
    model_config = ConfigDict(extra='forbid')


class PitStopForeignKeys(data_import.PitStopForeignKeys, SessionValidatorMixin):
    model_config = ConfigDict(extra='forbid')


class TeamReferenceValidatorMixin:
    @model_validator(mode='before')
    @classmethod
    def get_team_reference(cls, data: dict) -> dict:
        if data['year'] in TEAMS:
            if data['team_reference'] in TEAMS[data['year']]:
                data['team_reference'] = TEAMS[data['year']][data['team_reference']]
                return data
            else:
                raise ValueError(
                    f"team {data['team_reference']} not found in year {data['year']}'s team name "
                    f"mapping. Available teams: {TEAMS[data['year']].keys()}"
                )
        else:
            raise ValueError(f'year {data["year"]} not supported. Available years: {TEAMS.keys()}')


class RoundEntryForeignKeys(data_import.RoundEntryForeignKeys, TeamReferenceValidatorMixin):
    model_config = ConfigDict(extra='forbid')


class TeamDriverForeignKeys(data_import.TeamDriverForeignKeys, TeamReferenceValidatorMixin):
    model_config = ConfigDict(extra='forbid')
