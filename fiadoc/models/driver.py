# -*- coding: utf-8 -*-
"""Driver entry models"""
from jolpica.schemas import data_import
from pydantic import ConfigDict

from .foreign_key import RoundEntry


class RoundEntryObject(data_import.RoundEntryObject):
    model_config = ConfigDict(extra='forbid')


class RoundEntryImport(data_import.RoundEntryImport):
    foreign_keys: RoundEntry
    objects: list[RoundEntryObject]

    model_config = ConfigDict(extra='forbid')


class DriverObject(data_import.DriverObject):
    model_config = ConfigDict(extra='forbid')


class DriverImport(data_import.F1ImportSchema):
    foreign_keys: data_import.DriverForeignKeys = data_import.DriverForeignKeys()
    objects: list[DriverObject]

    model_config = ConfigDict(extra='forbid')
