import math
import os
from pathlib import Path
import time
import cv2
import streamlit as st
import tempfile
import pandas as pd

from pages.edit_frame import detect_and_render_frame
from tools.find_paths import (
    all_movement,
    create_tracked_objects,
    find_all_uar_sequences,
    find_longest_a2b_sequence,
)
from tools.process_objects import identify_umpire, process_objects, score_batter_runners
from tools.object_track_types import (
    BasesData,
    BasesEntry,
    DeepsortOutput,
    MovementSequence,
    SimpleTrackedObject,
    SimpleTrackedObjects,
    TrackingData,
)
from tools.utils import (
    distance_between_points,
    get_first_and_last_points,
    get_last_video_frame_index,
    get_np_points,
)


def build_tracking_data(
    deepsort_file: str,
    bases_data: BasesEntry,
    home_tolerance: float,
    auto_home_tolerance: bool,
    first_tolerance: float,
    frame,
) -> TrackingData:
    deepsort_output: DeepsortOutput
    with open(deepsort_file, "r") as f:
        deepsort_output = DeepsortOutput.model_validate_json(f.read())
    tracked_objects = create_tracked_objects(deepsort_output)
    umpire_id = -1
    if auto_home_tolerance:
        rich_tracked_objects = process_objects(deepsort_output)
        umpire_scores, umpire_id = identify_umpire(
            deepsort_output, rich_tracked_objects, (frame.shape[1], frame.shape[0])
        )
        avg_area = rich_tracked_objects.objects[umpire_id].avg_bbox_area or 0
        # home tolerance is such that the area of the circle defined by a radius = home tolerance
        # is equal to the avg area of the umpire bounding boxes
        # compute the radius of the circle from the area
        home_tolerance = math.sqrt(avg_area / math.pi)
        # home_tolerance = sqrt(avg_area) if avg_area is not None else home_tolerance
        print(f"Auto home tolerance: {home_tolerance}")

    longest_a2b_sequences: list[tuple[SimpleTrackedObject, MovementSequence]] = []
    for obj in tracked_objects.objects.values():
        seq = find_longest_a2b_sequence(
            obj=obj,
            step=1,
            pt_a=bases_data.home_pos,
            pt_b=bases_data.first_pos,
            pt_a_tolerance=home_tolerance,
            pt_b_tolerance=first_tolerance,
        )
        if seq is not None:
            longest_a2b_sequences.append((obj, seq))

    full_sequences: list[tuple[SimpleTrackedObject, MovementSequence]] = [
        (obj, all_movement(obj, step=1)[0]) for obj in tracked_objects.objects.values()
    ]

    return TrackingData(
        deepsort_output=deepsort_output,
        longest_a2b_sequences=longest_a2b_sequences,
        full_sequences=full_sequences,
        tracked_objects=tracked_objects,
        home_tolerance=int(home_tolerance),
        umpire_id=umpire_id,
    )


st.set_page_config(layout="wide")
# Initialize the YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use the appropriate YOLOv8 model weights

# Set Streamlit layout
st.sidebar.title("Select and Play Video")

# Directory containing the videos
video_dir = Path("ungitable/video")

# List all .mp4 files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]

# Select video file
selected_video = st.sidebar.radio("Choose a video...", video_files)
# selected_video = st.sidebar.selectbox(
#     "Choose a video...", video_files, format_func=lambda x: x
# )

# load bases as a list of BaseEntry
bases: BasesData
with open("ungitable/bases.json", "r") as bases_file:
    bases = BasesData.model_validate_json(bases_file.read())

# create a dictionary where each key is the file field from bases.entries
# and the value is the BasesEntry object
bases_dict = {entry.file: entry for entry in bases.entries}


# Set up mouse callback
selected_identities = []
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

if selected_video is not None:
    video = cv2.VideoCapture(str(video_dir / selected_video))

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    st.sidebar.write(f"Frames per second (FPS): {fps}")
    st.sidebar.write(f"Total frames: {total_frames}")
    st.sidebar.write(f"Duration (seconds): {duration:.2f}")

    frame_cols = st.columns([1, 1, 10], vertical_alignment="center")

    # Initialize Streamlit slider for frame navigation
    with frame_cols[0]:
        if st.button("<<"):
            st.session_state.frame_idx = max(0, st.session_state.frame_idx - 10)
    with frame_cols[1]:
        if st.button("\>\>"):  # type: ignore
            st.session_state.frame_idx = min(
                total_frames - 1, st.session_state.frame_idx + 10
            )
    with frame_cols[2]:
        st.session_state.frame_idx = st.slider(
            label="Frame",
            min_value=0,
            max_value=total_frames - 1,
            step=1,
            value=st.session_state.frame_idx,
        )
    tolerance_cols = st.columns([1, 0.5, 1.5], vertical_alignment="center")
    with tolerance_cols[0]:
        home_tolerance = st.slider("Home Plate Tolerance", 0, 400, 150, 5)
    with tolerance_cols[1]:
        auto_home_tolerance = st.checkbox("Auto Home Tolerance")

    with tolerance_cols[2]:
        first_tolerance = st.slider("First Base Tolerance", 0, 400, 25, 5)

    filter_cols = st.columns([1, 1], vertical_alignment="center")
    # get a comma separated list of filtered identities to display
    # from a streamlit text input
    filtered_identities = []
    with filter_cols[0]:
        filter_text = st.text_input("Filter Identities", "")
        filtered_identities = [
            int(x) for x in filter_text.split(",") if x.strip().isdigit()
        ]

    # checkbox whether to display bounding boxes
    with filter_cols[1]:
        display_bounding_boxes = st.checkbox("Display Bounding Boxes", value=True)
        home_to_first = st.checkbox("Home to First", value=True)

    video_frame_placeholder = st.empty()

    # Play/Pause functionality

    # def is_inside_tolerances(object_path: tuple[SimpleTrackedObject, MovementSequence]):
    #     obj = object_path[0]
    #     first_point, last_point = get_first_and_last_points(object_path)
    #     home_pos = bases_dict[selected_video].home_pos
    #     first_pos = bases_dict[selected_video].first_pos
    #     home_distance = distance_between_points(home_pos, first_point)
    #     first_distance = distance_between_points(first_pos, last_point)
    #     return home_distance < home_tolerance and first_distance < first_tolerance

    # Set video to the frame at the slider position
    video.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
    success, frame = video.read()

    deepsort_output_path = f"{video_dir / selected_video}.deepsort.json"

    if "tracking_data" not in st.session_state:
        st.session_state.tracking_data = {"key": "uninitialized", "value": None}

    current_key = f"{deepsort_output_path}:home_tolerance:{home_tolerance}:first_tolerance:{first_tolerance}:auto_home_tolerance:{auto_home_tolerance}"
    if st.session_state.tracking_data["key"] != current_key:
        st.session_state.tracking_data["key"] = current_key
        st.session_state.tracking_data["value"] = build_tracking_data(
            deepsort_file=deepsort_output_path,
            bases_data=bases_dict[selected_video],
            home_tolerance=home_tolerance,
            auto_home_tolerance=auto_home_tolerance,
            first_tolerance=first_tolerance,
            frame=frame,
        )

    tracking_data: TrackingData = st.session_state.tracking_data["value"]
    filtered_sequences = (
        tracking_data.longest_a2b_sequences
        if home_to_first
        else tracking_data.full_sequences
    )
    deepsort_output = tracking_data.deepsort_output
    tracked_objects = tracking_data.tracked_objects
    print(f"filtered_sequences: {len(filtered_sequences)}")

    # filtered_sequences = uar_sequences

    # get the top score and index

    seq_df = pd.DataFrame(
        [
            {
                "index": i,
                "identifer": seq[0].identity,
                "first_frame": seq[1].initial_video_frame,
                "last_frame": get_last_video_frame_index(seq),
            }
            for i, seq in enumerate(filtered_sequences)
        ]
    )
    # Add a selection checkbox column
    seq_df["Select"] = False

    # Display the editable DataFrame
    edited_df = st.data_editor(
        seq_df,
        column_config={"Select": st.column_config.CheckboxColumn("Select")},
        hide_index=True,
        use_container_width=True,
    )

    # Filter selected items
    selected_items = edited_df[edited_df["Select"] == True]
    if not selected_items.empty:
        selected_indexes = selected_items["index"].to_list()
    else:
        selected_indexes = []

    frame = detect_and_render_frame(
        frame=frame,
        frame_idx=st.session_state.frame_idx,
        uars=filtered_sequences,
        deepsort_output=deepsort_output,
        selected_indexes=selected_indexes,
        home_tolerance=tracking_data.home_tolerance,
        first_tolerance=first_tolerance,
        bases_dict=bases_dict,
        selected_video=selected_video,
        filtered_identities=filtered_identities,
        display_bounding_boxes=display_bounding_boxes,
        tracked_objects=tracked_objects,
        umpire_identity=tracking_data.umpire_id,
    )

    # Display the annotated frame
    try:
        video_frame_placeholder.image(frame, channels="BGR", caption="Detected Objects")
    except Exception as e:
        st.write(f"Error rendering image: {e}")

    # Release the video when done
    video.release()
else:
    st.write("Please upload a video file to start.")
