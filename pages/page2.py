import os
import cv2
import streamlit as st
import tempfile

# Initialize the YOLOv8 model
# model = YOLO("yolov8n.pt")  # Use the appropriate YOLOv8 model weights

# Set Streamlit layout
st.title("Video Object Detection with YOLOv8")
st.sidebar.title("Select and Play Video")

# Directory containing the videos
video_dir = 'ungitable/video'

# List all .mp4 files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

# Select video file
selected_video = st.sidebar.selectbox("Choose a video...", video_files)

def detect_and_render_frame(frame, video_frame_placeholder):
    # results = model(frame)

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


if selected_video:
    video_path = os.path.join(video_dir, selected_video)
    video = cv2.VideoCapture(video_path)

    fps = int(video.get(cv2.CAP_PROP_FPS))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))