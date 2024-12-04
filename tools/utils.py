import math
from typing import Callable
import numpy as np

from tools.object_track_types import (
    MovementSequence,
    SimpleTrackedObject,
    SumMovementSequence,
    TrackedObject,
    TrackedObjectFrame,
)


def intersection_area(boxA: list[float], boxB: list[float]) -> float:
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # return the intersection area
    return interArea


def baseline_midpoint(bbox: list[float]) -> list[float]:
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, max(y1, y2)]


def distance_between_points(p1: list[float], p2: list[float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def vector_magnitude(v: list[float]) -> float:
    return math.sqrt(v[0] ** 2 + v[1] ** 2)


def vector_between_points(p1: list[float], p2: list[float]) -> list[float]:
    return [p2[0] - p1[0], p2[1] - p1[1]]


def distance_between_bboxes(bbox1: list[float], bbox2: list[float]) -> float:
    return distance_between_points(baseline_midpoint(bbox1), baseline_midpoint(bbox2))


def vector_between_bboxes(bbox1: list[float], bbox2: list[float]) -> list[float]:
    return vector_between_points(baseline_midpoint(bbox1), baseline_midpoint(bbox2))


def calculate_travel_distance(tracked_object: TrackedObject) -> float:
    if len(tracked_object.frames) < 2:
        return 0.0
    distances = [
        vector_magnitude(frame.vector_from_prev_frame or [0, 0])
        for frame in tracked_object.frames
    ]
    return sum(distances)


def is_bbox_contained_in_bbox(bbox1: list[float], bbox2: list[float]) -> bool:
    return (
        bbox1[0] >= bbox2[0]
        and bbox1[1] >= bbox2[1]
        and bbox1[2] <= bbox2[2]
        and bbox1[3] <= bbox2[3]
    )


def calculate_travel_bbox(tracked_object: TrackedObject) -> list[float]:
    if len(tracked_object.frames) < 2:
        return []
    # walk over each frame, get the baseline midpoint. Create a bbox from the min and max of the midpoints
    midpoints = [baseline_midpoint(frame.bbox_xyxy) for frame in tracked_object.frames]
    min_x = min([midpoint[0] for midpoint in midpoints])
    max_x = max([midpoint[0] for midpoint in midpoints])
    min_y = min([midpoint[1] for midpoint in midpoints])
    max_y = max([midpoint[1] for midpoint in midpoints])
    return [min_x, min_y, max_x, max_y]


def calculate_avg_bbox_area(tracked_object: TrackedObject) -> float:
    if len(tracked_object.frames) < 1:
        return 0.0
    areas = [
        (frame.bbox_xyxy[2] - frame.bbox_xyxy[0])
        * (frame.bbox_xyxy[3] - frame.bbox_xyxy[1])
        for frame in tracked_object.frames
    ]
    return sum(areas) / len(areas)


# return a movement sequence for the longest sequence of frames
# for which the predicate is true. For the sum, use the sum of the ValueGen callable
def longest_sequence(
    frames: list[TrackedObjectFrame],
    start_index: int,
    predicate: Callable[[TrackedObjectFrame], bool],
    value_gen: Callable[[TrackedObjectFrame], float],
) -> SumMovementSequence:
    longest_sequence = 0
    current_sequence = 0
    current_start = 0
    longest_start = 0
    current_sum = 0.0
    longest_sum = 0.0
    for i, frame in enumerate(frames[start_index:]):
        if predicate(frame):
            current_sequence += 1
            current_sum += value_gen(frame)
            if current_sequence > longest_sequence:
                longest_sequence = current_sequence
                longest_start = current_start
                longest_sum = current_sum
        else:
            current_sequence = 0
            current_start = i
            current_sum = 0.0
    return SumMovementSequence(
        initial_video_frame=frames[longest_start].video_frame_index,
        initial_object_frame=longest_start,
        count=longest_sequence,
        final_object_frame=longest_start + longest_sequence - 1,
        final_video_frame=frames[
            longest_start + longest_sequence - 1
        ].video_frame_index,
        sum=longest_sum,
    )


def all_sequences(
    frames: list[TrackedObjectFrame],
    step: int,
    predicate: Callable[[TrackedObjectFrame, TrackedObjectFrame], bool],
) -> list[MovementSequence]:
    sequences = []
    current_sequence = 0
    current_start = 0
    for i in range(0, len(frames) - step, step):
        frame = frames[i]
        next_frame = frames[i + step]
        if predicate(frame, next_frame):
            current_sequence += step
        else:
            if current_sequence > 0:
                sequences.append(
                    MovementSequence(
                        initial_video_frame=frames[current_start].video_frame_index,
                        initial_object_frame=current_start,
                        count=current_sequence,
                        final_object_frame=current_start + current_sequence - 1,
                        final_video_frame=frames[
                            current_start + current_sequence - 1
                        ].video_frame_index,
                    )
                )
            current_sequence = 0
            current_start = i
    if current_sequence > 0:
        sequences.append(
            MovementSequence(
                initial_video_frame=frames[current_start].video_frame_index,
                initial_object_frame=current_start,
                count=current_sequence,
                final_object_frame=current_start + current_sequence - 1,
                final_video_frame=frames[
                    current_start + current_sequence - 1
                ].video_frame_index,
            )
        )
    return sequences


def frame_moves_right(
    frame: TrackedObjectFrame, next_frame: TrackedObjectFrame
) -> bool:
    return vector_between_bboxes(frame.bbox_xyxy, next_frame.bbox_xyxy)[0] > 0.0


def frame_moves_left(frame: TrackedObjectFrame, next_frame: TrackedObjectFrame) -> bool:
    return vector_between_bboxes(frame.bbox_xyxy, next_frame.bbox_xyxy)[0] < 0.0


def frame_moves_up(frame: TrackedObjectFrame, next_frame: TrackedObjectFrame) -> bool:
    return vector_between_bboxes(frame.bbox_xyxy, next_frame.bbox_xyxy)[1] < 0.0


def frame_moves_down(frame: TrackedObjectFrame, next_frame: TrackedObjectFrame) -> bool:
    return vector_between_bboxes(frame.bbox_xyxy, next_frame.bbox_xyxy)[1] > 0.0


def frame_moves_right_or_up(
    frame: TrackedObjectFrame, next_frame: TrackedObjectFrame
) -> bool:
    return (
        frame_moves_right(frame, next_frame) and not frame_moves_down(frame, next_frame)
    ) or (frame_moves_up(frame, next_frame) and not frame_moves_left(frame, next_frame))


# def find_tracked_object_frame_index_for_video_frame_index(
#     tracked_object: SimpleTrackedObject, video_frame_index: int
# ) -> int:
#     for i, frame in enumerate(tracked_object.frames):
#         if frame.video_frame_index == video_frame_index:
#             return i
#     return -1


# def get_first_and_last_frame_indexes(
#     object_path: tuple[SimpleTrackedObject, MovementSequence]
# ):
#     obj = object_path[0]
#     movement_sequence = object_path[1]
#     first_tof = find_tracked_object_frame_index_for_video_frame_index(
#         obj, movement_sequence.initial_video_frame
#     )
#     last_tof = first_tof + movement_sequence.count - 1
#     return first_tof, last_tof


# def get_last_video_frame_index(
#     object_path: tuple[SimpleTrackedObject, MovementSequence]
# ):
#     first_tof, last_tof = get_first_and_last_frame_indexes(object_path)
#     return object_path[0].frames[last_tof].video_frame_index


def get_first_and_last_points(
    object_path: tuple[SimpleTrackedObject, MovementSequence]
):
    obj = object_path[0]
    seq = object_path[1]
    return (
        baseline_midpoint(obj.frames[seq.initial_object_frame].bbox_xyxy),
        baseline_midpoint(obj.frames[seq.final_object_frame].bbox_xyxy),
    )


def get_np_points(object_path: tuple[SimpleTrackedObject, MovementSequence]):
    obj = object_path[0]
    movement_sequence = object_path[1]
    first_tof = movement_sequence.initial_object_frame
    all_points = [
        baseline_midpoint(frame.bbox_xyxy)
        for frame in obj.frames[first_tof : first_tof + movement_sequence.count]
    ]
    # return a numpy array of dtype int32 in shape (n, 1, 2)
    return np.array(
        [[[int(point[0]), int(point[1])]] for point in all_points], dtype=np.int32
    )
