# -*- coding: utf-8 -*-
"""
{
    "object": "pit_stop",
    "foreign_keys": {
        "season_year": 2023,
        "round_number": 22,
        "type": "R",
        "car_number": 1,
        "lap": 16
    },
    "object": [
        {
            "number": 1,
            "duration": timedelta(seconds=21, milliseconds=662),
            "local_timestamp": "17:27:36"
        },
        ...
    ]
}
"""

from datetime import timedelta

from jolpica.schemas import data_import
from pydantic import BaseModel, ConfigDict, PositiveInt

from .foreign_key import PitStopForeignKeys


class PitStopObject(data_import.PitStopObject):
    duration: dict[str, str | int]

    model_config = ConfigDict(extra="forbid")


class PitStopData(
    data_import.PitStopImport
):  # TODO: all xxxData can be combined into one class?
    foreign_keys: PitStopForeignKeys
    objects: list[PitStopObject]

    model_config = ConfigDict(extra="forbid")
