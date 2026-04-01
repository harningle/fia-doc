# -*- coding: utf-8 -*-
import numpy as np
from jolpica.schemas import data_import
from pydantic import ConfigDict, field_validator

from .foreign_key import SessionEntryForeignKeys


class LapObject(data_import.LapObject):
    time: dict[str, str | int]  # TODO: what's the time here??

    model_config = ConfigDict(extra='forbid')

    @field_validator('number', mode='before')
    @classmethod
    def _nan_to_none(cls, v):
        if isinstance(v, float) and np.isnan(v):
            return None
        return v


class LapImport(data_import.LapImport):
    foreign_keys: SessionEntryForeignKeys
    objects: list[LapObject]

    model_config = ConfigDict(extra='forbid')
