# -*- coding: utf-8 -*-
from datetime import timedelta

from pydantic import BaseModel, ConfigDict

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
