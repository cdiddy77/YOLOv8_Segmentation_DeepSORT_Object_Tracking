import os
from pathlib import Path
import time
import cv2
import streamlit as st
import tempfile

from tools.process_objects import process_objects, score_objects
from tools.object_track_types import DeepsortOutput

print("LOADING PAGE 2")
# Initialize the YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use the appropriate YOLOv8 model weights

# Set Streamlit layout
st.title("Video Object Detection with YOLOv8")
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


def detect_and_render_frame(
    frame,
    video_frame_placeholder,
    frame_idx: int,
    identities: list[int],
    winning_identity: int,
    deepsort_output: DeepsortOutput,
):
    print(f"Frame index: {frame_idx}")
    # results = model(frame)
    # get the deepsort output frame
    for identity in identities:
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

                color = (0, 255, 0) if identity != winning_identity else (255, 0, 0)
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
            else:
                print(f"Identity {identity} not found in frame {frame_idx}")
    # Draw rectangles around detected objects
    # for box in results[0].boxes:
    #     x1, y1, x2, y2 = box.xyxy[0]  # Coordinates of the box
    #     conf = box.conf[0]  # Confidence
    #     cls = box.cls[0]  # Class

    #     # Draw rectangle and label on frame
    #     frame = cv2.rectangle(
    #         frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2
    #     )
    #     frame = cv2.putText(
    #         frame,
    #         f"{model.names[int(cls)]} {conf:.2f}",
    #         (int(x1), int(y1) - 10),
    #         cv2.FONT_HERSHEY_SIMPLEX,
    #         0.5,
    #         (0, 255, 0),
    #         2,
    #     )

    # Display the annotated frame
    video_frame_placeholder.image(frame, channels="BGR", caption="Detected Objects")


if selected_video is not None:
    print(selected_video)
    video = cv2.VideoCapture(str(video_dir / selected_video))

    deepsort_output_path = f"{video_dir / selected_video}.deepsort.json"
    deepsort_output: DeepsortOutput
    with open(deepsort_output_path, "r") as f:
        deepsort_output = DeepsortOutput.model_validate_json(f.read())
    tracked_objects = process_objects(deepsort_output)
    scores = score_objects(deepsort_output, tracked_objects)
    print(scores.model_dump_json(indent=2))

    # get the top score and index
    top_x, top_score = max(scores.overall.items(), key=lambda x: x[1])

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    st.sidebar.write(f"Frames per second (FPS): {fps}")
    st.sidebar.write(f"Total frames: {total_frames}")
    st.sidebar.write(f"Duration (seconds): {duration:.2f}")
    st.sidebar.write(f"Top scoring object: {top_x}({top_score})")

    # Initialize Streamlit slider for frame navigation
    frame_idx = st.slider("Frame", 0, total_frames - 1, 0)
    video_frame_placeholder = st.empty()

    # Play/Pause functionality
    is_playing = st.sidebar.button("Play/Pause")
    if "playing" not in st.session_state:
        st.session_state.playing = False  # Initialize play state

    # Toggle play/pause when the button is clicked
    if is_playing:
        st.session_state.playing = not st.session_state.playing

    # Set video to the frame at the slider position
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    success, frame = video.read()

    if st.session_state.playing:
        while st.session_state.playing:
            # Calculate the frame to show based on FPS and time elapsed
            current_time = frame_idx / fps
            frame_idx += fps  # Move forward 1 second in the video

            # Ensure the frame index doesn't go beyond the total frame count
            if frame_idx >= total_frames:
                frame_idx = 0
                st.session_state.playing = False  # Stop when video reaches the end
                break

            # Set the video to the next frame
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = video.read()

            if success:
                # Display the current frame
                # st.image(frame, channels="BGR")
                detect_and_render_frame(
                    frame,
                    video_frame_placeholder,
                    frame_idx,
                    list(scores.overall.keys()),
                    top_x,
                    deepsort_output,
                )
            else:
                st.write("Unable to read frame from video.")
                break

            # Sleep to simulate real-time playback
            time.sleep(1)
    else:
        detect_and_render_frame(
            frame,
            video_frame_placeholder,
            frame_idx,
            list(scores.overall.keys()),
            top_x,
            deepsort_output,
        )

    # Release the video when done
    video.release()
else:
    st.write("Please upload a video file to start.")
