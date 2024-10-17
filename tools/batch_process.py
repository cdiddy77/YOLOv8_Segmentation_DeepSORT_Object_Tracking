import os
import subprocess

# Define the directory containing the videos
video_dir = 'ungitable/video'
segment_dir = 'ultralytics/yolo/v8/segment'

# Get the absolute path of the video directory
video_dir_abs = os.path.abspath(video_dir)

# Iterate over all mp4 files in the video directory
for video_file in os.listdir(video_dir_abs):
    if video_file.endswith('.mp4'):
        video_path = os.path.join(video_dir_abs, video_file)
        deepsort_output_path = f"{video_path}.deepsort.json"
        
        # Construct the command
        command = (
            f"cd {segment_dir} && "
            f"python predict.py source=\"{video_path}\" model=yolov8x-seg.pt "
            # f"max_frames=100 "
            f"deepsort_outputs_filename=\"{deepsort_output_path}\""
        )
        print(f"Executing command: \n{command}")
        # Execute the command and show the output
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        print(f"Output for {video_file}:")
        print(stdout.decode())
        if stderr:
            print(f"Errors for {video_file}:")
            print(stderr.decode())