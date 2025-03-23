# -*- coding: utf-8 -*-
from typing import Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator
)

from .foreign_key import SessionEntryForeignKeys
from jolpica.schemas import data_import


class SessionEntryObject(data_import.SessionEntryObject):
    time: dict[str, str | int] | None

    model_config = ConfigDict(extra='forbid')

class SessionEntryImport(data_import.SessionEntryImport):
    object_type: str = 'SessionEntry'
    foreign_keys: SessionEntryForeignKeys
    objects: list[SessionEntryObject]

    model_config = ConfigDict(extra='forbid')


class QualiClassification(BaseModel):
    position: PositiveInt
    is_classified: bool


class QualiClassificationData(BaseModel):
    object_type: str = 'SessionEntry'
    foreign_keys: SessionEntryForeignKeys
    objects: list[QualiClassification]

    model_config = ConfigDict(extra='forbid')
