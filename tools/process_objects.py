# parse args, the name of the input file and the name of the output file
import argparse
import json

from tools.object_track_types import (
    DeepsortOutput,
    HeuristicalScores,
    SimpleTrackedObjects,
    TrackedObject,
    TrackedObjectFrame,
    TrackedObjects,
    UmpireScores,
)
from tools.utils import (
    calculate_avg_bbox_area,
    calculate_travel_bbox,
    calculate_travel_distance,
    intersection_area,
    is_bbox_contained_in_bbox,
    longest_sequence,
    vector_between_bboxes,
)


def process_objects(deepsort_output: DeepsortOutput):
    tracked_objects: TrackedObjects = TrackedObjects(
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
        bat_indexes = [i for i, id in enumerate(frame.object_id) if id == bat_index]
        bat_bboxes: list[list[float]] = [frame.bbox_xyxy[x] for x in bat_indexes]
        # for each object
        for bbox, identity, object_id in zip(
            frame.bbox_xyxy, frame.identities, frame.object_id
        ):
            if object_id != person_index:
                continue

            # find the intersection area with the bat
            bat_intersect_area = 0.0
            for bat_bbox in bat_bboxes:
                bat_intersect_area += intersection_area(bbox, bat_bbox)
            # print the bounding box, identity, and object id
            if identity not in tracked_objects.objects:
                tracked_objects.objects[identity] = TrackedObject(
                    identity=identity,
                    object_id=object_id,
                    frames=[],
                    tags=[],
                    travel_distance=None,
                    travel_bbox=None,
                    total_bat_intersect_area=None,
                    count_bat_intersect_area=None,
                    longest_rightward_sequence=None,
                    longest_leftward_sequence=None,
                    avg_bbox_area=None,
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
                    bat_intersect_area=bat_intersect_area,
                    vector_from_prev_frame=vector_from_prev_frame,
                    missing_frames=missing_frames,
                )
            )

    # calculate various stats for each tracked object
    for tracked_object in tracked_objects.objects.values():
        tracked_object.travel_distance = calculate_travel_distance(tracked_object)
        tracked_object.travel_bbox = calculate_travel_bbox(tracked_object)
        tracked_object.total_bat_intersect_area = sum(
            [frame.bat_intersect_area or 0.0 for frame in tracked_object.frames]
        )
        tracked_object.count_bat_intersect_area = len(
            [frame for frame in tracked_object.frames if frame.bat_intersect_area]
        )

        tracked_object.longest_rightward_sequence = longest_sequence(
            frames=tracked_object.frames,
            start_index=0,
            predicate=lambda frame: frame.vector_from_prev_frame[0] >= 0.0,
            value_gen=lambda frame: frame.vector_from_prev_frame[0],
        )
        tracked_object.longest_leftward_sequence = longest_sequence(
            frames=tracked_object.frames,
            start_index=0,
            predicate=lambda frame: frame.vector_from_prev_frame[0] <= 0.0,
            value_gen=lambda frame: frame.vector_from_prev_frame[0],
        )
        tracked_object.avg_bbox_area = calculate_avg_bbox_area(tracked_object)

    return tracked_objects


def identify_umpire(
    deepsort_output: DeepsortOutput,
    tracked_objects: TrackedObjects,
    dims: tuple[int, int],
) -> tuple[UmpireScores, int]:
    frame_count = len(deepsort_output.frames)
    TOP_N = 3
    left_rect = [0.0, 0, float(dims[0] / 6), float(dims[1])]
    right_rect = [float(dims[0] * 5 / 6), 0, float(dims[0]), float(dims[1])]
    # biggest and lowest object in the frame
    scorable_objects = [
        v
        for v in tracked_objects.objects.values()
        # only consider if dont have a travel bbox that is entirely in the left or right quarter of
        # the screen, since that would probably be the SL scoreboard "card"
        if (
            v.travel_bbox
            and not is_bbox_contained_in_bbox(v.travel_bbox, left_rect)
            and not is_bbox_contained_in_bbox(v.travel_bbox, right_rect)
        )
    ]
    bottom_of_view_first_frame = sorted(
        [
            v
            for v in scorable_objects
            # only consider objects that appear in the first quarter of the video
            if v.frames[0].video_frame_index < frame_count / 4
        ],
        key=lambda obj: obj.frames[0].bbox_xyxy[3] if obj.frames else 0.0,
        reverse=True,
    )[:TOP_N]

    greatest_avg_bbox_area = sorted(
        scorable_objects,
        key=lambda obj: obj.avg_bbox_area or 0.0,
        reverse=True,
    )[:TOP_N]

    scores: UmpireScores = UmpireScores(
        bottom_of_view_first_frame={},
        avg_bbox_area={},
        overall={},
    )

    for obj in bottom_of_view_first_frame:
        scores.bottom_of_view_first_frame[obj.identity] = (
            obj.frames[0].bbox_xyxy[3] if obj.frames else 0.0
        )
    for obj in greatest_avg_bbox_area:
        scores.avg_bbox_area[obj.identity] = obj.avg_bbox_area or 0.0

    sum_bottom_of_view_first_frame = max(
        sum(scores.bottom_of_view_first_frame.values()), 0.01
    )
    for tracked_object in bottom_of_view_first_frame:
        id = tracked_object.identity
        scores.overall[id] = scores.overall.get(id, 0.0) + (
            scores.bottom_of_view_first_frame[id] / sum_bottom_of_view_first_frame
        )
    sum_avg_bbox_area = max(sum(scores.avg_bbox_area.values()), 0.01)
    for tracked_object in greatest_avg_bbox_area:
        id = tracked_object.identity
        scores.overall[id] = scores.overall.get(id, 0.0) + (
            scores.avg_bbox_area[id] / sum_avg_bbox_area
        )
    return scores, max(scores.overall.items(), key=lambda x: x[1])[0]


def score_batter_runners(
    deepsort_output: DeepsortOutput, tracked_objects: TrackedObjects, exclude: list[int]
) -> HeuristicalScores:
    frame_count = len(deepsort_output.frames)

    scorable_objects = [
        obj for obj in tracked_objects.objects.values() if obj.identity not in exclude
    ]
    # take the top 3 of each stat. Assign each identity a weighted score based on their rank in each stat
    TOP_N = 3
    farthest_travelers = sorted(
        scorable_objects,
        key=lambda obj: obj.travel_distance or 0.0,
        reverse=True,
    )[:TOP_N]
    max_bat_intersects = sorted(
        scorable_objects,
        key=lambda obj: obj.total_bat_intersect_area or 0.0,
        reverse=True,
    )[:TOP_N]
    farthest_consecutive_rights = sorted(
        scorable_objects,
        key=lambda obj: (
            obj.longest_rightward_sequence.sum
            if obj.longest_rightward_sequence
            else 0.0
        ),
        reverse=True,
    )[:TOP_N]

    bottom_of_view_first_frame = sorted(
        [
            v
            for v in scorable_objects
            if v.frames[0].video_frame_index < frame_count / 4
        ],
        key=lambda obj: obj.frames[0].bbox_xyxy[3] if obj.frames else 0.0,
        reverse=True,
    )[:TOP_N]

    greatest_avg_bbox_area = sorted(
        scorable_objects,
        key=lambda obj: obj.avg_bbox_area or 0.0,
        reverse=True,
    )[:TOP_N]

    weights = {
        "farthest_travelers": 0,  # not included in score
        "max_bat_intersects": 1,
        "farthest_consecutive_rights": 1,
        "bottom_of_view_first_frame": 1,
        "avg_bbox_area": 0,
    }
    scores: HeuristicalScores = HeuristicalScores(
        farthest_travelers={},
        max_bat_intersects={},
        farthest_consecutive_rights={},
        bottom_of_view_first_frame={},
        avg_bbox_area={},
        overall={},
    )
    for obj in farthest_travelers:
        scores.farthest_travelers[obj.identity] = obj.travel_distance or 0.0
    for obj in max_bat_intersects:
        scores.max_bat_intersects[obj.identity] = obj.total_bat_intersect_area or 0.0
    for obj in farthest_consecutive_rights:
        scores.farthest_consecutive_rights[obj.identity] = (
            (
                obj.longest_rightward_sequence.sum
                if obj.longest_rightward_sequence
                else 0.0
            ),
            (
                obj.longest_rightward_sequence.initial_video_frame
                if obj.longest_rightward_sequence
                else -1
            ),
        )
    for obj in bottom_of_view_first_frame:
        scores.bottom_of_view_first_frame[obj.identity] = (
            obj.frames[0].bbox_xyxy[3] if obj.frames else 0.0
        )
    for obj in greatest_avg_bbox_area:
        scores.avg_bbox_area[obj.identity] = obj.avg_bbox_area or 0.0

    # compute overall score for each identity based on the weighted sum of the individual scores
    # individual scores are a percentage of the total score for top N
    sum_farthest_travelers = max(sum(scores.farthest_travelers.values()), 0.01)
    for tracked_object in farthest_travelers:
        id = tracked_object.identity
        scores.overall[id] = scores.overall.get(id, 0.0) + (
            (scores.farthest_travelers[id] / sum_farthest_travelers)
            * weights["farthest_travelers"]
        )
    sum_max_bat_intersects = max(sum(scores.max_bat_intersects.values()), 0.01)
    for tracked_object in max_bat_intersects:
        id = tracked_object.identity
        scores.overall[id] = scores.overall.get(id, 0.0) + (
            (scores.max_bat_intersects[id] / sum_max_bat_intersects)
            * weights["max_bat_intersects"]
        )
    sum_farthest_consecutive_rights = max(
        sum([x[0] for x in scores.farthest_consecutive_rights.values()]), 0.01
    )
    for tracked_object in farthest_consecutive_rights:
        id = tracked_object.identity
        scores.overall[id] = scores.overall.get(id, 0.0) + (
            (
                scores.farthest_consecutive_rights[id][0]
                / sum_farthest_consecutive_rights
            )
            * weights["farthest_consecutive_rights"]
        )
    sum_bottom_of_view_first_frame = max(
        sum(scores.bottom_of_view_first_frame.values()), 0.01
    )
    for tracked_object in bottom_of_view_first_frame:
        id = tracked_object.identity
        scores.overall[id] = scores.overall.get(id, 0.0) + (
            (scores.bottom_of_view_first_frame[id] / sum_bottom_of_view_first_frame)
            * weights["bottom_of_view_first_frame"]
        )
    sum_avg_bbox_area = max(sum(scores.avg_bbox_area.values()), 0.01)
    for tracked_object in greatest_avg_bbox_area:
        id = tracked_object.identity
        scores.overall[id] = scores.overall.get(id, 0.0) + (
            (scores.avg_bbox_area[id] / sum_avg_bbox_area) * weights["avg_bbox_area"]
        )
    return scores
    # analytics: HeuristicalAnalytics = HeuristicalAnalytics(
    #     farthest_traveler=farthest_traveler.identity,
    #     max_bat_intersect=max_bat_intersect.identity,
    #     farthest_consecutive_right=farthest_consecutive_right.identity,
    # )
    # print(analytics.model_dump_json(indent=2))


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

    tracked_objects = process_objects(deepsort_output)
    if args.output:
        with open(args.output, "w") as f:
            f.write(tracked_objects.model_dump_json(indent=2))
    else:
        print(tracked_objects.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
