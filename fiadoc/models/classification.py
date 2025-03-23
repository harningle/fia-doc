# -*- coding: utf-8 -*-
from pydantic import (
    ConfigDict,
)

from .foreign_key import SessionEntryForeignKeys
from jolpica.schemas import data_import


class SessionEntryObject(data_import.SessionEntryObject):
    time: dict[str, str | int] | None = None

    model_config = ConfigDict(extra='forbid')

class SessionEntryImport(data_import.SessionEntryImport):
    object_type: str = 'SessionEntry'
    foreign_keys: SessionEntryForeignKeys
    objects: list[SessionEntryObject]

    model_config = ConfigDict(extra='forbid')
