import os
import subprocess

# Define the directory containing the videos
video_dir = "ungitable/video"
segment_dir = "ultralytics/yolo/v8/segment"

# Get the absolute path of the video directory
video_dir_abs = os.path.abspath(video_dir)

# Iterate over all mp4 files in the video directory
# for video_file in os.listdir(video_dir_abs):
# for video_file in ["6cCuJiEJBNTx8bHz.mp4", "Q9zX9ZTN9nKe2gy0.mp4"]:
for video_file in ["7v1aqMSmJ59buyhy.mp4", "7ysTo5G80DevGa3H.mp4"]:
    if video_file.endswith(".mp4"):
        video_path = os.path.join(video_dir_abs, video_file)
        # deepsort_output_path = f"{video_path}.deepsort.json"
        deepsort_output_path = f"{video_path}.deepsort3.json"

        # compare file timestamps. If the deepsort output file is newer
        # than the video file, skip the video
        if os.path.exists(deepsort_output_path):
            video_ts = os.path.getmtime(video_path)
            deepsort_ts = os.path.getmtime(deepsort_output_path)
            if deepsort_ts > video_ts:
                print(
                    f"Deepsort output file is newer than video file. Skipping {video_file}"
                )
                continue

        # Construct the command
        command = (
            f"cd {segment_dir} && "
            f'python predict.py source="{video_path}" model=yolov8x-seg.pt '
            # f"max_frames=100 "
            f'deepsort_outputs_filename="{deepsort_output_path}"'
        )
        print(f"Executing command: \n{command}")
        # Execute the command and show the output
        process = subprocess.Popen(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = process.communicate()

        print(f"Output for {video_file}:")
        print(stdout.decode())
        if stderr:
            print(f"Errors for {video_file}:")
            print(stderr.decode())
