# -*- coding: utf-8 -*-
from pydantic import BaseModel, ConfigDict, NonNegativeFloat, NonNegativeInt, PositiveInt

from .foreign_key import SessionEntry


class Classification(BaseModel):
    position: PositiveInt
    is_classified: bool
    status: NonNegativeInt
    points: NonNegativeFloat
    time: PositiveInt               # TODO: what's the value for drivers crash on lap 1?
    fastest_lap: PositiveInt        # TODO: Should allow missing here?
    fastest_lap_rank: PositiveInt
    laps_completed: NonNegativeInt  # TODO: or positive int? What if a driver retires in lap 1?


class ClassificationData(BaseModel):
    object_type: str = 'SessionEntry'
    foreign_keys: SessionEntry
    objects: list[Classification]

    model_config = ConfigDict(extra='forbid')


class QualiClassification(BaseModel):
    position: PositiveInt
    fastest_lap: PositiveInt  # I don't think this is the right way. Now we have quali laps, so
                              # this should link to the lap time object


class QualiClassificationData(BaseModel):
    object_type: str = 'SessionEntry'
    foreign_keys: SessionEntry
    objects: list[QualiClassification]

    model_config = ConfigDict(extra='forbid')
