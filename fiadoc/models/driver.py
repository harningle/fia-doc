# -*- coding: utf-8 -*-
"""Driver entry models"""
from pydantic import BaseModel, ConfigDict, PositiveInt

from .foreign_key import RoundEntry
from jolpica.schemas import data_import


class RoundEntryObject(data_import.RoundEntryObject):
    model_config = ConfigDict(extra='forbid')


class RoundEntryImport(data_import.RoundEntryImport):
    foreign_keys: RoundEntry
    objects: list[RoundEntryObject]

    model_config = ConfigDict(extra='forbid')
