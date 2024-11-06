# parse args, the name of the input file and the name of the output file
import argparse
import json
from typing import Callable

from tools.object_track_types import (
    DeepsortOutput,
    HeuristicalScores,
    MovementSequence,
    SimpleTrackedObject,
    SimpleTrackedObjects,
    TrackedObject,
    TrackedObjectFrame,
    TrackedObjects,
    UmpireScores,
)
from tools.utils import (
    all_sequences,
    baseline_midpoint,
    calculate_avg_bbox_area,
    calculate_travel_bbox,
    calculate_travel_distance,
    distance_between_points,
    frame_moves_right_or_up,
    intersection_area,
    is_bbox_contained_in_bbox,
    longest_sequence,
    vector_between_bboxes,
)


def create_tracked_objects(deepsort_output: DeepsortOutput) -> SimpleTrackedObjects:
    tracked_objects: SimpleTrackedObjects = SimpleTrackedObjects(
        objects={}, object_id_names=deepsort_output.object_id_names
    )

    person_index = [
        i for i, name in enumerate(deepsort_output.object_id_names) if name == "person"
    ][0]
    bat_index = [
        i
        for i, name in enumerate(deepsort_output.object_id_names)
        if name == "baseball bat"
    ][0]

    # process all the frames and create a set of tracked objects, each with a set of frames
    for ndx, frame in enumerate(deepsort_output.frames):
        # find each object that is a baseball bat, grab the bounding box
        # bat_indexes = [i for i, id in enumerate(frame.object_id) if id == bat_index]
        # bat_bboxes: list[list[float]] = [frame.bbox_xyxy[x] for x in bat_indexes]
        # for each object
        for bbox, identity, object_id in zip(
            frame.bbox_xyxy, frame.identities, frame.object_id
        ):
            if object_id != person_index:
                continue

            # find the intersection area with the bat
            # bat_intersect_area = 0.0
            # for bat_bbox in bat_bboxes:
            #     bat_intersect_area += intersection_area(bbox, bat_bbox)
            # print the bounding box, identity, and object id
            if identity not in tracked_objects.objects:
                tracked_objects.objects[identity] = SimpleTrackedObject(
                    identity=identity,
                    object_id=object_id,
                    frames=[],
                )

            # get previous frame
            prevFrame: TrackedObjectFrame | None = (
                tracked_objects.objects[identity].frames[-1]
                if len(tracked_objects.objects[identity].frames) > 0
                else None
            )
            missing_frames = ndx - prevFrame.video_frame_index - 1 if prevFrame else 0
            vector_from_prev_frame = (
                vector_between_bboxes(prevFrame.bbox_xyxy, bbox)
                if prevFrame
                else [0.0, 0.0, 0.0, 0.0]
            )

            tracked_objects.objects[identity].frames.append(
                TrackedObjectFrame(
                    video_frame_index=ndx,
                    bbox_xyxy=bbox,
                    bat_intersect_area=0.0,  # we don't care for simple tracked objects
                    vector_from_prev_frame=vector_from_prev_frame,
                    missing_frames=missing_frames,
                )
            )
    return tracked_objects


def find_all_uar_sequences(
    tracked_objects: SimpleTrackedObjects, step: int
) -> list[tuple[SimpleTrackedObject, MovementSequence]]:
    result: list[tuple[SimpleTrackedObject, MovementSequence]] = []
    for identity, tracked_object in tracked_objects.objects.items():
        sequences = all_sequences(
            frames=tracked_object.frames,
            step=step,
            predicate=frame_moves_right_or_up,
        )
        for sequence in sequences:
            result.append((tracked_object, sequence))
    return result


def all_movement(object: SimpleTrackedObject, step: int):
    sequences = all_sequences(
        frames=object.frames,
        step=step,
        predicate=lambda frame, next_frame: True,
    )
    if len(sequences) == 0:
        return [
            MovementSequence(
                initial_video_frame=object.frames[0].video_frame_index, count=0
            )
        ]
    return sequences


def find_longest_a2b_sequence(
    obj: SimpleTrackedObject,
    step: int,
    pt_a: list[float],
    pt_b: list[float],
    pt_a_tolerance: float,
    pt_b_tolerance: float,
) -> MovementSequence | None:
    """
    Find the longest sequence where the start of the sequence is within a
    tolerance of point A and the end of the sequence is within a tolerance of point B
    """
    result: MovementSequence | None = None
    # go until you find a point within the tolerance of point A
    current_start = 0
    while current_start < len(obj.frames):
        while (
            current_start < len(obj.frames)
            and distance_between_points(
                pt_a, baseline_midpoint(obj.frames[current_start].bbox_xyxy)
            )
            > pt_a_tolerance
        ):
            current_start += step
        # if you found a point within the tolerance of point A
        if current_start < len(obj.frames):
            # start from the current start and go until you find a point within the tolerance of point B
            current_end = current_start
            while (
                current_end < len(obj.frames)
                and distance_between_points(
                    pt_b, baseline_midpoint(obj.frames[current_end].bbox_xyxy)
                )
                > pt_b_tolerance
            ):
                current_end += step
            # set the result to the longest sequence found
            if current_end < len(obj.frames):
                result = MovementSequence(
                    initial_video_frame=obj.frames[current_start].video_frame_index,
                    count=current_end - current_start,
                )
                # continue searching for a longer sequence
                while (
                    current_end < len(obj.frames)
                    and distance_between_points(
                        pt_b, baseline_midpoint(obj.frames[current_end].bbox_xyxy)
                    )
                    < pt_b_tolerance
                ):
                    current_end += step
                    result.count = current_end - current_start
            current_start = current_end

    return result


def first_frame_of_sequence(
    obj: SimpleTrackedObject,
    sequence: MovementSequence,
    predicate: Callable[[TrackedObjectFrame], bool],
):
    for i in range(
        sequence.initial_video_frame, sequence.initial_video_frame + sequence.count
    ):
        if predicate(obj.frames[i]):
            return i
    return None


# write the output file
def main():
    parser = argparse.ArgumentParser(description="Process deepsort output")

    # required
    parser.add_argument("--input", type=str, help="input file", required=True)
    parser.add_argument("--output", type=str, help="output file")

    args = parser.parse_args()

    deepsort_output: DeepsortOutput
    with open(args.input, "r") as f:
        deepsort_output = DeepsortOutput.model_validate_json(f.read())

    tracked_objects = create_tracked_objects(deepsort_output)
    if args.output:
        with open(args.output, "w") as f:
            f.write(tracked_objects.model_dump_json(indent=2))
    else:
        print(tracked_objects.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
