# -*- coding: utf-8 -*-
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    field_validator
)

from .foreign_key import SessionEntry


class Classification(BaseModel):
    position: PositiveInt
    is_classified: bool
    status: NonNegativeInt
    points: NonNegativeFloat
    time: dict[str, str | int]
    laps_completed: NonNegativeInt  # TODO: or positive int? What if retire in lap 1?
    fastest_lap_rank: PositiveInt   # TODO: what if DNS or retire in lap 1?

    model_config = ConfigDict(extra='forbid')


class ClassificationData(BaseModel):
    object_type: str = 'SessionEntry'
    foreign_keys: SessionEntry
    objects: list[Classification]

    model_config = ConfigDict(extra='forbid')


class QualiClassification(BaseModel):
    position: PositiveInt


class QualiClassificationData(BaseModel):
    object_type: str = 'SessionEntry'
    foreign_keys: SessionEntry
    objects: list[QualiClassification]

    model_config = ConfigDict(extra='forbid')
