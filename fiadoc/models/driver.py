# -*- coding: utf-8 -*-
"""Driver entry models"""
from jolpica.schemas import data_import
from pydantic import ConfigDict

from .foreign_key import RoundEntryForeignKeys, TeamDriverForeignKeys


class RoundEntryObject(data_import.RoundEntryObject):
    model_config = ConfigDict(extra='forbid')


class RoundEntryImport(data_import.RoundEntryImport):
    foreign_keys: RoundEntryForeignKeys
    objects: list[RoundEntryObject]

    model_config = ConfigDict(extra='forbid')


class DriverObject(data_import.DriverObject):
    model_config = ConfigDict(extra='forbid')


class DriverImport(data_import.F1ImportSchema):
    object_type: str = 'Driver'
    foreign_keys: data_import.DriverForeignKeys = data_import.DriverForeignKeys()
    objects: list[DriverObject]

    model_config = ConfigDict(extra='forbid')


class TeamDriverObject(data_import.TeamDriverObject):
    model_config = ConfigDict(extra='forbid')


class TeamDriverImport(data_import.TeamDriverImport):
    foreign_keys: TeamDriverForeignKeys
    objects: list[TeamDriverObject]

    model_config = ConfigDict(extra='forbid')
