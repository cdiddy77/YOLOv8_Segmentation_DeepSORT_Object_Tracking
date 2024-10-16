# parse args, the name of the input file and the name of the output file
import argparse

from tools.object_track_types import (
    DeepsortOutput,
    HeuristicalAnalytics,
    TrackedObject,
    TrackedObjectFrame,
    TrackedObjects,
)
from tools.utils import (
    calculate_travel_bbox,
    calculate_travel_distance,
    distance_between_bboxes,
    intersection_area,
    longest_sequence,
    vector_between_bboxes,
)


parser = argparse.ArgumentParser(description="Process deepsort output")

# required
parser.add_argument("--input", type=str, help="input file", required=True)
parser.add_argument("--output", type=str, help="output file")

args = parser.parse_args()

deepsort_output: DeepsortOutput
# read the input file into a dict
with open(args.input, "r") as f:
    deepsort_output = DeepsortOutput.model_validate_json(f.read())

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

frame_count = len(deepsort_output.frames)

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
        predicate=lambda frame: frame.vector_from_prev_frame[0] >= 0.0,
        value_gen=lambda frame: frame.vector_from_prev_frame[0],
    )
    tracked_object.longest_leftward_sequence = longest_sequence(
        frames=tracked_object.frames,
        predicate=lambda frame: frame.vector_from_prev_frame[0] <= 0.0,
        value_gen=lambda frame: frame.vector_from_prev_frame[0],
    )

farthest_travelers = sorted(
    tracked_objects.objects.values(), key=lambda obj: obj.travel_distance or 0.0
)
farthest_traveler = max(
    tracked_objects.objects.values(), key=lambda obj: obj.travel_distance or 0.0
)

max_bat_intersects = sorted(
    tracked_objects.objects.values(),
    key=lambda obj: obj.total_bat_intersect_area or 0.0,
)
max_bat_intersect = max(
    tracked_objects.objects.values(),
    key=lambda obj: obj.total_bat_intersect_area or 0.0,
)

farthest_consecutive_rights = sorted(
    tracked_objects.objects.values(),
    key=lambda obj: (
        obj.longest_rightward_sequence.sum if obj.longest_rightward_sequence else 0.0
    ),
)
farthest_consecutive_right = max(
    tracked_objects.objects.values(),
    key=lambda obj: (
        obj.longest_rightward_sequence.sum if obj.longest_rightward_sequence else 0.0
    ),
)

bottom_of_view = sorted(
    tracked_objects.objects.values(),
    key=lambda obj: obj.travel_bbox[3] if obj.travel_bbox else 0.0,
)

bottom_of_view_first_frame = sorted(
    [
        v
        for v in tracked_objects.objects.values()
        if v.frames[0].video_frame_index < frame_count / 4
    ],
    key=lambda obj: obj.frames[0].bbox_xyxy[3] if obj.frames else 0.0,
)

analytics: HeuristicalAnalytics = HeuristicalAnalytics(
    farthest_traveler=farthest_traveler.identity,
    max_bat_intersect=max_bat_intersect.identity,
    farthest_consecutive_right=farthest_consecutive_right.identity,
)
print(analytics.model_dump_json(indent=2))

# write the output file
if args.output:
    with open(args.output, "w") as f:
        f.write(tracked_objects.model_dump_json(indent=2))
else:
    print(tracked_objects.model_dump_json(indent=2))
