import os
from pathlib import Path
import cv2
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from pages.edit_frame import detect_and_render_frame
from tools.find_paths import (
    build_clip_prediction,
    build_tracking_data,
)
from tools.object_track_types import (
    BasesData,
    TrackingData,
)
from tools.process_objects import measure_movement

st.set_page_config(layout="wide")
# Initialize the YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use the appropriate YOLOv8 model weights

# Set Streamlit layout
st.sidebar.title("Select and Play Video")

# Directory containing the videos
video_dir = Path("ungitable/video")

# List all .mp4 files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
video_files.sort()
print(video_files[:10])

# Select video file
selected_video = st.sidebar.radio("Choose a video...", video_files)
# selected_video = st.sidebar.selectbox(
#     "Choose a video...", video_files, format_func=lambda x: x
# )

# Set up mouse callback
selected_identities = []
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

if selected_video is not None:
    video = cv2.VideoCapture(str(video_dir / selected_video))

    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    st.sidebar.write(f"Frames per second (FPS): {fps}")
    st.sidebar.write(f"Total frames: {total_frames}")
    st.sidebar.write(f"Duration (seconds): {duration:.2f}")

    frame_cols = st.columns([1, 1, 10, 1], vertical_alignment="center")
    # movement_chart_placeholder = st.empty()
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
        movement_chart_placeholder = st.empty()

    with frame_cols[3]:
        st.write(f"ts: {st.session_state.frame_idx / fps:.2f}")
    tolerance_cols = st.columns([1, 0.5, 1.5], vertical_alignment="center")
    with tolerance_cols[0]:
        home_tolerance = st.slider("Home Plate Tolerance", 0, 400, 150, 5)
    with tolerance_cols[1]:
        auto_home_tolerance = st.checkbox("Auto Home Tolerance", value=True)

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
        cb_cols = st.columns([1, 1, 1], vertical_alignment="center")
        with cb_cols[0]:
            display_bounding_boxes = st.checkbox("Display Bounding Boxes", value=True)
        with cb_cols[1]:
            home_to_first = st.checkbox("Home to First", value=True)
        with cb_cols[2]:
            a2b_or_exit = st.checkbox("A2B or Exit", value=True)

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

    # get the video width and height
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    deepsort_output_path = f"{video_dir / selected_video}.deepsort.json"
    annotation_path = f"{video_dir / selected_video.split('.')[0]}.annotation.json"

    if "tracking_data" not in st.session_state:
        st.session_state.tracking_data = {"key": "uninitialized", "value": None}

    current_key = f"{deepsort_output_path}:home_tolerance:{home_tolerance}:first_tolerance:{first_tolerance}:auto_home_tolerance:{auto_home_tolerance}:a2b_or_exit:{a2b_or_exit}"
    if st.session_state.tracking_data["key"] != current_key:
        st.session_state.tracking_data["key"] = current_key
        st.session_state.tracking_data["value"] = build_tracking_data(
            deepsort_file=deepsort_output_path,
            annotation_file=annotation_path,
            home_radius=home_tolerance,
            auto_home_radius=auto_home_tolerance,
            first_radius=first_tolerance,
            frame_width=frame_width,
            frame_height=frame_height,
        )

    tracking_data: TrackingData = st.session_state.tracking_data["value"]
    filtered_sequences = (
        (
            tracking_data.longest_a2b_sequences
            if a2b_or_exit
            else tracking_data.longest_exiting_sequences
        )
        if home_to_first
        else tracking_data.full_sequences
    )
    deepsort_output = tracking_data.deepsort_output
    tracked_objects = tracking_data.tracked_objects
    print(f"filtered_sequences: {len(filtered_sequences)}")
    clip_prediction = build_clip_prediction(tracking_data, fps)
    print(f"clip_prediction: {clip_prediction}")

    raw_df = pd.DataFrame(
        {
            "Frame": range(1, len(deepsort_output.frames)),
            "Movement": tracking_data.movement,
        }
    )

    # make a dataframe which has movement aggregated by batches of 10 frames and the
    # first column is the time in seconds
    batch_size = int(fps)
    batched_df = pd.DataFrame(
        {
            "Time": [
                round(i / fps, 2)
                for i in range(0, len(deepsort_output.frames) - batch_size, batch_size)
            ],
            "Movement": [
                sum(tracking_data.movement[i : i + batch_size])
                for i in range(0, len(deepsort_output.frames) - batch_size, batch_size)
            ],
        }
    )
    movement_chart_placeholder.bar_chart(
        batched_df, x="Time", y="Movement", height=200, y_label=""
    )

    # filtered_sequences = uar_sequences

    # get the top score and index

    seq_df = pd.DataFrame(
        [
            {
                "index": i,
                "identifer": seq[0].identity,
                "first_frame": seq[1].initial_video_frame,
                "last_frame": seq[1].final_video_frame,
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
        annotation=tracking_data.annotation,
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
