import math
from typing import Callable

from tools.object_track_types import MovementSequence, TrackedObject, TrackedObjectFrame


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


# return a movement sequence for the longest sequence of frames
# for which the predicate is true. For the sum, use the sum of the ValueGen callable
def longest_sequence(
    frames: list[TrackedObjectFrame],
    predicate: Callable[[TrackedObjectFrame], bool],
    value_gen: Callable[[TrackedObjectFrame], float],
) -> MovementSequence:
    longest_sequence = 0
    current_sequence = 0
    current_start = 0
    longest_start = 0
    current_sum = 0.0
    longest_sum = 0.0
    for i, frame in enumerate(frames):
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
    return MovementSequence(
        initial_frame=frames[longest_start].video_frame_index,
        count=longest_sequence,
        sum=longest_sum,
    )
