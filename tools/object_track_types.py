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


class VideoExtraData(BaseModel):
    home_plate: list[float] = Field(
        ..., description="Home plate coordinates in format [x, y]"
    )
    first_base: list[float] = Field(
        ..., description="First base coordinates in format [x, y]"
    )


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


class BasesEntry(BaseModel):
    file: str = Field(..., description="File name")
    home_pos_xy: list[float] = Field(
        ..., description="Home plate position", alias="homePos"
    )
    first_pos_xy: list[float] = Field(
        ..., description="First base position", alias="firstPos"
    )


class BasesData(BaseModel):
    entries: list[BasesEntry] = Field(..., description="List of bases entries")


class Annotation(BaseModel):
    home_pos_xy: list[float] = Field(
        ..., description="Home plate position", alias="homePos"
    )
    first_pos_xy: list[float] = Field(
        ..., description="First base position", alias="firstPos"
    )
    contact_time: Optional[float] = Field(
        description="Contact time",
        alias="contactTime",
        default=None,
    )


class MovementSequence(BaseModel):
    initial_video_frame: int = Field(..., description="Initial frame of the sequence")
    initial_object_frame: int = Field(
        ..., description="Initial object frame of the sequence"
    )
    count: int = Field(..., description="Number of frames in the sequence")
    final_object_frame: int = Field(
        ..., description="Final object frame of the sequence"
    )
    final_video_frame: int = Field(..., description="Final frame of the sequence")


class SumMovementSequence(MovementSequence):
    sum: float = Field(
        ..., description="Sum of the directional movement in the sequence"
    )


class SimpleTrackedObject(BaseModel):
    identity: int = Field(..., description="Identity of the object")
    object_id: int = Field(..., description="Object id")
    frames: list[TrackedObjectFrame] = Field(
        ..., description="List of frames with bounding boxes"
    )


class SimpleTrackedObjects(BaseModel):
    objects: dict[int, SimpleTrackedObject] = Field(
        ..., description="List of tracked objects"
    )
    object_id_names: list[str] = Field(..., description="List of object id names")


class TrackedObject(SimpleTrackedObject):
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
    longest_rightward_sequence: Optional[SumMovementSequence] = Field(
        ..., description="Longest rightward sequence"
    )
    longest_leftward_sequence: Optional[SumMovementSequence] = Field(
        ..., description="Longest leftward sequence"
    )
    avg_bbox_area: Optional[float] = Field(..., description="Average bounding box area")


class TrackedObjects(BaseModel):
    objects: dict[int, TrackedObject] = Field(
        ..., description="List of tracked objects"
    )
    object_id_names: list[str] = Field(..., description="List of object id names")


class IdValue(BaseModel):
    id: int = Field(..., description="Identity")
    value: float = Field(..., description="Value")


class UmpireScores(BaseModel):
    bottom_of_view_first_frame: dict[int, float] = Field(
        ..., description="top n bottom of view first frame"
    )
    avg_bbox_area: dict[int, float] = Field(
        ..., description="top n average bounding box area"
    )
    overall: dict[int, float] = Field(
        ..., description="Overall score for each identity"
    )


class HeuristicalScores(BaseModel):
    farthest_travelers: dict[int, float] = Field(
        ..., description="top n farthest travelers"
    )
    max_bat_intersects: dict[int, float] = Field(
        ..., description="top n max bat intersects"
    )
    farthest_consecutive_rights: dict[int, tuple[float, int]] = Field(
        ..., description="top n farthest consecutive rights"
    )
    bottom_of_view_first_frame: dict[int, float] = Field(
        ..., description="top n bottom of view first frame"
    )
    avg_bbox_area: dict[int, float] = Field(
        ..., description="top n average bounding box area"
    )
    overall: dict[int, float] = Field(
        ..., description="Overall score for each identity"
    )


class TrackingData(BaseModel):
    deepsort_output: DeepsortOutput = Field(..., description="Deepsort output")
    annotation: Annotation = Field(..., description="Annotation")
    longest_a2b_sequences: list[tuple[SimpleTrackedObject, MovementSequence]] = Field(
        ..., description="Longest A to B movement sequences"
    )
    longest_exiting_sequences: list[tuple[SimpleTrackedObject, MovementSequence]] = (
        Field(..., description="Longest exiting movement sequences")
    )
    full_sequences: list[tuple[SimpleTrackedObject, MovementSequence]] = Field(
        ..., description="Full movement sequences"
    )
    tracked_objects: SimpleTrackedObjects = Field(..., description="Tracked objects")
    home_tolerance: int = Field(..., description="Home tolerance")
    umpire_id: int = Field(..., description="Umpire identity")
    movement: list[float] = Field(..., description="Movement vector")


class ObjectTrackingPrediction(BaseModel):
    contact_moment: float = Field(..., description="Contact moment")
    event_type: Literal["hit", "no-hit", "unknown"] = Field(
        ..., description="Event type"
    )
