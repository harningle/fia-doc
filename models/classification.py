# -*- coding: utf-8 -*-
from datetime import timedelta

from pydantic import BaseModel, ConfigDict, PositiveInt

from models.foreign_key import SessionEntry


class Classification(BaseModel):
    points: float
    time: timedelta
    laps_completed: int  # TODO: or positive int? What if a driver retires in lap 1?


class ClassificationData(BaseModel):
    object_type: str = 'classification'
    foreign_keys: SessionEntry
    objects: list[Classification]

    model_config = ConfigDict(extra='forbid')


class QualiClassification(BaseModel):
    position: PositiveInt
    fastest_lap: timedelta  # I don't think this is the right way. Now we have quali laps, so this
                            # should link to the lap time object


class QualiClassificationData(BaseModel):
    object_type: str = 'classification'
    foreign_keys: SessionEntry
    objects: list[QualiClassification]

    model_config = ConfigDict(extra='forbid')
