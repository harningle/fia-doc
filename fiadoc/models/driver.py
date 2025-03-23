# -*- coding: utf-8 -*-
"""Driver entry models"""

from jolpica.schemas import data_import
from pydantic import BaseModel, ConfigDict, PositiveInt

from .foreign_key import RoundEntry


class RoundEntryObject(data_import.RoundEntryObject):
    model_config = ConfigDict(extra="forbid")


class RoundEntryImport(data_import.RoundEntryImport):
    foreign_keys: RoundEntry
    objects: list[RoundEntryObject]

    model_config = ConfigDict(extra="forbid")
