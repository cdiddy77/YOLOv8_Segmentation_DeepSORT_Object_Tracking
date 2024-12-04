import cv2
from tools.object_track_types import (
    Annotation,
    BasesEntry,
    DeepsortOutput,
    MovementSequence,
    SimpleTrackedObject,
    SimpleTrackedObjects,
)
from tools.utils import get_np_points

DARK_TEAL = (90, 120, 60)
BURGUNDY = (49, 0, 98)
ORANGE = (0, 165, 255)
MUSTARD_YELLOW = (34, 162, 255)
DEEP_PURPLE = (76, 0, 153)
CORAL = (80, 127, 255)
SOFT_PINK = (203, 192, 255)
OLIVE_GREEN = (45, 85, 60)
SKY_BLUE = (235, 206, 135)
CRIMSON = (60, 20, 220)
LIME_GREEN = (50, 205, 50)
SLATE_GRAY = (144, 128, 112)
TURQUOISE = (208, 224, 64)


def detect_and_render_frame(
    frame,
    frame_idx: int,
    uars: list[tuple[SimpleTrackedObject, MovementSequence]],
    selected_indexes: list[int],
    deepsort_output: DeepsortOutput,
    tracked_objects: SimpleTrackedObjects,
    home_tolerance: int,
    first_tolerance: int,
    annotation: Annotation,
    filtered_identities: list[int],
    display_bounding_boxes: bool,
    umpire_identity: int,
):
    if display_bounding_boxes:
        box_identities = (
            filtered_identities
            if filtered_identities
            else [identity for identity in tracked_objects.objects.keys()]
        )
        for identity in box_identities:
            if frame_idx >= len(deepsort_output.frames):
                print(
                    f"Frame index {frame_idx} exceeds the number of frames in the deepsort output"
                )
            else:
                deepsort_frame = deepsort_output.frames[frame_idx]
                # find the index of the identity
                if identity in deepsort_frame.identities:
                    identity_index = deepsort_frame.identities.index(identity)
                    bbox = deepsort_frame.bbox_xyxy[identity_index]

                    color = LIME_GREEN if identity == umpire_identity else TURQUOISE

                    frame = cv2.rectangle(
                        frame,
                        (int(bbox[0]), int(bbox[1])),
                        (int(bbox[2]), int(bbox[3])),
                        color,
                        2,
                    )
                    frame = cv2.putText(
                        frame,
                        f"{identity}",
                        (int(bbox[0]), int(bbox[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )
                # else:
                #     print(f"Identity {identity} not found in frame {frame_idx}")
    # draw a circle of radius home_tolerance around home plate
    home_pos = annotation.home_pos_xy
    first_pos = annotation.first_pos_xy
    frame = cv2.circle(
        frame,
        (int(home_pos[0]), int(home_pos[1])),
        home_tolerance,
        ORANGE,
        thickness=2,
    )
    frame = cv2.circle(
        frame,
        (int(first_pos[0]), int(first_pos[1])),
        first_tolerance,
        ORANGE,
        thickness=2,
    )

    # results = model(frame)
    # get the deepsort output frame
    for ndx, uar in enumerate(uars):
        obj = uar[0]
        movement_sequence = uar[1]
        identity = obj.identity
        if not filtered_identities or identity in filtered_identities:
            if frame_idx >= len(deepsort_output.frames):
                print(
                    f"Frame index {frame_idx} exceeds the number of frames in the deepsort output"
                )
                continue

            pts = get_np_points(uar)
            color = DARK_TEAL if ndx in selected_indexes else DEEP_PURPLE
            # iterate over the points, draw a dot for each and a line between them
            frame = cv2.polylines(
                frame, [pts], isClosed=False, color=color, thickness=2
            )
    return frame
