from typing import Literal, Optional
from pydantic import BaseModel, Field


class DeepsortOutputFrame(BaseModel):
    bbox_xyxy: list[list[float]] = Field(
        ..., description="Bounding box coordinates in format [x1, y1, x2, y2]"
    )
    identities: list[int] = Field(
        ..., description="List of identities for each bounding box"
    )
    object_id: list[int] = Field(
        ..., description="List of object ids for each bounding box"
    )


class DeepsortOutput(BaseModel):
    frames: list[DeepsortOutputFrame] = Field(
        ..., description="List of frames with bounding boxes and identities"
    )
    object_id_names: list[str] = Field(..., description="List of object id names")


class TrackedObjectFrame(BaseModel):
    video_frame_index: int = Field(
        ..., description="Frame index from the original video"
    )
    bbox_xyxy: list[float] = Field(
        ..., description="Bounding box coordinates in format [x1, y1, x2, y2]"
    )
    bat_intersect_area: float = Field(
        ..., description="Bounding box area intersection with bat"
    )
    vector_from_prev_frame: list[float] = Field(
        ..., description="Vector from previous frame"
    )

    missing_frames: int = Field(..., description="Number of missing frames")


# literal for all the different types of tags
Tags = Literal["farthest_traveler", "max_bat_intersect", "farthest_right"]


class MovementSequence(BaseModel):
    initial_frame: int = Field(..., description="Initial frame of the sequence")
    count: int = Field(..., description="Number of frames in the sequence")
    sum: float = Field(
        ..., description="Sum of the directional movement in the sequence"
    )


class TrackedObject(BaseModel):
    identity: int = Field(..., description="Identity of the object")
    object_id: int = Field(..., description="Object id")
    frames: list[TrackedObjectFrame] = Field(
        ..., description="List of frames with bounding boxes"
    )
    tags: list[Tags] = Field(..., description="List of tags")
    travel_distance: Optional[float] = Field(..., description="Travel distance")
    travel_bbox: Optional[list[float]] = Field(
        ..., description="Travel bounding box xyxy"
    )
    total_bat_intersect_area: Optional[float] = Field(
        ..., description="Total bat intersection area"
    )
    count_bat_intersect_area: Optional[int] = Field(
        ..., description="Number of bat intersections"
    )
    longest_rightward_sequence: Optional[MovementSequence] = Field(
        ..., description="Longest rightward sequence"
    )
    longest_leftward_sequence: Optional[MovementSequence] = Field(
        ..., description="Longest leftward sequence"
    )


class TrackedObjects(BaseModel):
    objects: dict[int, TrackedObject] = Field(
        ..., description="List of tracked objects"
    )
    object_id_names: list[str] = Field(..., description="List of object id names")


class HeuristicalAnalytics(BaseModel):
    farthest_traveler: int = Field(..., description="Identity of the farthest traveler")
    max_bat_intersect: int = Field(..., description="Identity of the max bat intersect")
    farthest_consecutive_right: int = Field(
        ..., description="Identity of the object moving farthest consecutive right"
    )
