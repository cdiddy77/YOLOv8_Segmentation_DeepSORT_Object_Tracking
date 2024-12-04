# parse args, the name of the input file and the name of the output file
import argparse
import math
from typing import Callable

from tools.object_track_types import (
    Annotation,
    BasesEntry,
    ObjectTrackingPrediction,
    DeepsortOutput,
    MovementSequence,
    SimpleTrackedObject,
    SimpleTrackedObjects,
    TrackedObjectFrame,
    TrackingData,
)
from tools.process_objects import identify_umpire, measure_movement, process_objects
from tools.utils import (
    all_sequences,
    baseline_midpoint,
    distance_between_points,
    frame_moves_right_or_up,
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
                initial_video_frame=object.frames[0].video_frame_index,
                initial_object_frame=0,
                count=0,
                final_object_frame=0,
                final_video_frame=0,
            )
        ]
    return sequences


def find_longest_exiting_sequence(
    obj: SimpleTrackedObject,
    step: int,
    pt_a: list[float],
    pt_b: list[float],
    pt_a_tolerance: float,
    pt_b_tolerance: float,
) -> MovementSequence | None:
    """
    Find the longest sequence where the start of the sequence is within a
    tolerance of point A and the end of the sequence is outside of the
    point A tolerance in the direction of point B
    """
    result: MovementSequence | None = None

    # get the angle, relative to the x-axis, of the vector from point A to point B
    # a_b_vector_angle = math.atan2(pt_b[1] - pt_a[1], pt_b[0] - pt_a[0])
    # # convert to degrees
    # a_b_vector_angle_deg = math.degrees(a_b_vector_angle)
    # quad_horiz = pt_b[0] - pt_a[0]
    # quad_vert = pt_b[1] - pt_a[1]

    # go until you find a point within the tolerance of point A
    current_start = 0
    distance_a_to_b = distance_between_points(pt_a, pt_b)
    while current_start < len(obj.frames):
        while (
            current_start < len(obj.frames)
            and distance_between_points(
                pt_a, baseline_midpoint(obj.frames[current_start].bbox_xyxy)
            )
            > pt_a_tolerance
        ):
            current_start += step

        # now move current start forward until we exit the tolerance of point A in the direction of point B
        while current_start < len(obj.frames) and (
            distance_between_points(
                pt_a, baseline_midpoint(obj.frames[current_start].bbox_xyxy)
            )
            < pt_a_tolerance
            or distance_a_to_b
            < distance_between_points(
                pt_b, baseline_midpoint(obj.frames[current_start].bbox_xyxy)
            )
        ):
            current_start += step

        # if you found a point within the tolerance of point A
        if current_start < len(obj.frames):
            # start from the current start and go until you find a point within the tolerance of point B
            current_end = current_start
            closest_distance = distance_between_points(
                pt_b, baseline_midpoint(obj.frames[current_end].bbox_xyxy)
            )
            closest_index = current_end
            while (
                current_end < len(obj.frames)
                and distance_between_points(
                    pt_b, baseline_midpoint(obj.frames[current_end].bbox_xyxy)
                )
                > pt_b_tolerance
            ):
                current_end += step
                if current_end < len(obj.frames):
                    new_distance = distance_between_points(
                        pt_b, baseline_midpoint(obj.frames[current_end].bbox_xyxy)
                    )
                    if new_distance < closest_distance:
                        closest_distance = new_distance
                        closest_index = current_end

            # set the result to the longest sequence found
            if current_end < len(obj.frames):
                result = MovementSequence(
                    initial_video_frame=obj.frames[current_start].video_frame_index,
                    initial_object_frame=current_start,
                    count=current_end - current_start + 1,
                    final_object_frame=current_end,
                    final_video_frame=obj.frames[current_end].video_frame_index,
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
            else:  # if we didn't find a point within the tolerance of point B
                if result is None:
                    result = MovementSequence(
                        initial_video_frame=obj.frames[current_start].video_frame_index,
                        initial_object_frame=current_start,
                        count=closest_index - current_start + 1,
                        final_object_frame=closest_index,
                        final_video_frame=obj.frames[closest_index].video_frame_index,
                    )
                elif closest_index - current_start > result.count:
                    result.initial_video_frame = obj.frames[
                        current_start
                    ].video_frame_index
                    result.count = closest_index - current_start

            current_start = current_end

    return result


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
    distance_a_to_b = distance_between_points(pt_a, pt_b)
    while current_start < len(obj.frames):
        while (
            current_start < len(obj.frames)
            and distance_between_points(
                pt_a, baseline_midpoint(obj.frames[current_start].bbox_xyxy)
            )
            > pt_a_tolerance
        ):
            current_start += step

        # now move current start forward until we exit the tolerance of point A in the direction of point B
        while current_start < len(obj.frames) and (
            distance_between_points(
                pt_a, baseline_midpoint(obj.frames[current_start].bbox_xyxy)
            )
            < pt_a_tolerance
            or distance_a_to_b
            < distance_between_points(
                pt_b, baseline_midpoint(obj.frames[current_start].bbox_xyxy)
            )
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
                    initial_object_frame=current_start,
                    count=current_end - current_start + 1,
                    final_object_frame=current_end,
                    final_video_frame=obj.frames[current_end].video_frame_index,
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


def build_tracking_data(
    deepsort_file: str,
    annotation_file: str,
    home_radius: float,
    auto_home_radius: bool,
    first_radius: float,
    frame_width: int,
    frame_height: int,
) -> TrackingData:
    deepsort_output: DeepsortOutput
    with open(deepsort_file, "r") as f:
        deepsort_output = DeepsortOutput.model_validate_json(f.read())
    annotation: Annotation
    with open(annotation_file, "r") as f:
        annotation = Annotation.model_validate_json(f.read())
    tracked_objects = create_tracked_objects(deepsort_output)
    umpire_id = -1
    if auto_home_radius:
        rich_tracked_objects = process_objects(deepsort_output)
        umpire_scores, umpire_id = identify_umpire(
            deepsort_output, rich_tracked_objects, (frame_width, frame_height)
        )
        avg_area = rich_tracked_objects.objects[umpire_id].avg_bbox_area or 0
        # home tolerance is such that the area of the circle defined by a radius = home tolerance
        # is equal to the avg area of the umpire bounding boxes
        # compute the radius of the circle from the area
        home_radius = math.sqrt(avg_area / math.pi)
        # home_tolerance = sqrt(avg_area) if avg_area is not None else home_tolerance
        print(f"Auto home tolerance: {home_radius}")

    # seq_fn = find_longest_a2b_sequence if a2b_or_exit else find_longest_exiting_sequence

    longest_a2b_sequences: list[tuple[SimpleTrackedObject, MovementSequence]] = []
    longest_exiting_sequences: list[tuple[SimpleTrackedObject, MovementSequence]] = []
    for obj in tracked_objects.objects.values():
        seq = find_longest_a2b_sequence(
            obj=obj,
            step=1,
            pt_a=annotation.home_pos_xy,
            pt_b=annotation.first_pos_xy,
            pt_a_tolerance=home_radius,
            pt_b_tolerance=first_radius,
        )
        if seq is not None:
            longest_a2b_sequences.append((obj, seq))
        seq2 = find_longest_exiting_sequence(
            obj=obj,
            step=1,
            pt_a=annotation.home_pos_xy,
            pt_b=annotation.first_pos_xy,
            pt_a_tolerance=home_radius,
            pt_b_tolerance=first_radius,
        )
        if seq2 is not None:
            longest_exiting_sequences.append((obj, seq2))

    full_sequences: list[tuple[SimpleTrackedObject, MovementSequence]] = [
        (obj, all_movement(obj, step=1)[0]) for obj in tracked_objects.objects.values()
    ]
    movement = [
        measure_movement(deepsort_output, index)
        for index in range(1, len(deepsort_output.frames))
    ]

    return TrackingData(
        deepsort_output=deepsort_output,
        annotation=annotation,
        longest_a2b_sequences=longest_a2b_sequences,
        longest_exiting_sequences=longest_exiting_sequences,
        full_sequences=full_sequences,
        tracked_objects=tracked_objects,
        home_tolerance=int(home_radius),
        umpire_id=umpire_id,
        movement=movement,
    )


def build_clip_prediction(
    tracking_data: TrackingData, fps: float
) -> ObjectTrackingPrediction:
    if tracking_data.longest_a2b_sequences:
        obj, movement = min(
            tracking_data.longest_a2b_sequences,
            key=lambda x: x[1].initial_video_frame,
        )
        return ObjectTrackingPrediction(
            contact_moment=movement.initial_video_frame / fps, event_type="hit"
        )
    elif tracking_data.longest_exiting_sequences:
        obj, movement = min(
            tracking_data.longest_exiting_sequences,
            key=lambda x: x[1].initial_video_frame,
        )
        return ObjectTrackingPrediction(
            contact_moment=movement.initial_video_frame / fps, event_type="no-hit"
        )
    else:
        return ObjectTrackingPrediction(contact_moment=0, event_type="unknown")


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
