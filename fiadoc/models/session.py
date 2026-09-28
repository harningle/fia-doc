# -*- coding: utf-8 -*-
from jolpica.schemas import data_import
from pydantic import ConfigDict

from .foreign_key import SessionForeignKeys


class SessionObject(data_import.SessionObject):
    model_config = ConfigDict(extra='forbid')


class SessionImport(data_import.SessionImport):
    object_type: str = 'Session'
    foreign_keys: SessionForeignKeys
    objects: list[SessionObject]

    model_config = ConfigDict(extra='forbid')
