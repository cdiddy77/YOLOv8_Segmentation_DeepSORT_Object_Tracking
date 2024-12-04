from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field

from tools.object_track_types import ObjectTrackingPrediction


class ClipPrediction(BaseModel):
    prediction: Literal[
        "no-video", "no-deepsort", "no-base-info", "exception", "prediction"
    ] = Field(..., description="Prediction status")
    object_tracking_prediction: Optional[ObjectTrackingPrediction] = Field(
        None, description="Object tracking prediction"
    )
    exception: Optional[Any] = Field(None, description="Exception message")
    actual_est_contact: Optional[float] = Field(
        None, description="Actual estimated contact"
    )


def clip_predict(key: str, file_path: Path) -> ClipPrediction:
    # does video exist
    # does deepsort file exist
    # does annotations file exist, and have an entry for this video
    # if all exists, then predict
    # catch exceptions

    return ClipPrediction(
        prediction="prediction",
        object_tracking_prediction=None,
        exception=None,
        actual_est_contact=None,
    )
