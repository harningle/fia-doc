# -*- coding: utf-8 -*-
from pydantic import BaseModel, ConfigDict, PositiveInt

from .foreign_key import SessionEntryForeignKeys
from jolpica.schemas import data_import


class LapObject(data_import.LapObject):
    time: dict[str, str | int]
    model_config = ConfigDict(extra='forbid')


class LapImport(data_import.LapImport):
    foreign_keys: SessionEntryForeignKeys
    objects: list[LapObject]

    model_config = ConfigDict(extra='forbid')
