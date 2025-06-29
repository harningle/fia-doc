# -*- coding: utf-8 -*-
from jolpica.schemas import data_import
from pydantic import ConfigDict

from .foreign_key import SessionEntryForeignKeys


class LapObject(data_import.LapObject):
    time: dict[str, str | int]
    model_config = ConfigDict(extra="forbid")


class LapImport(data_import.LapImport):
    foreign_keys: SessionEntryForeignKeys
    objects: list[LapObject]

    model_config = ConfigDict(extra="forbid")
