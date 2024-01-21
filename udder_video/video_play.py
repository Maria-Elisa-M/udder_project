# First import library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2
# Import os.path for file path manipulation
import os.path
import rosbag
import pandas as pd
import random

inpath = r"video_files\videos_1"
filelist = os.listdir(r"video_files\videos_1")
random.shuffle(filelist)
video_data = pd.DataFrame(index = range(len(filelist)), columns = ["file", "score", "frame_num"])

def get_num_frames(indir, filename):
    topic = "/device_0/sensor_0/Depth_0/image/data"
    bag = rosbag.Bag(os.path.join(indir, filename), "r")
    nframes = int(bag.get_type_and_topic_info()[1][topic][1])
    return nframes

for f, file in enumerate(filelist):
    cow_num = file.split("_")[0]

    frame_num = get_num_frames(inpath, file)
    try:
        # Create pipeline
        pipeline = rs.pipeline()

        # Create a config object
        config = rs.config()

        # Tell config that we will use a recorded device from file to be used by the pipeline through playback.
        rs.config.enable_device_from_file(config, os.path.join(inpath, file))

        # Configure the pipeline to stream the depth stream
        # Change this parameters according to the recorded bag file resolution
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)

        # Start streaming from file
        pipeline.start(config)

        # Create opencv window to render image in
        cv2.namedWindow(cow_num, cv2.WINDOW_AUTOSIZE)

        # Create colorizer object
        colorizer = rs.colorizer()

        i = 0
        # Streaming loop
        while True:
            # Get frameset of depth
            frames = pipeline.wait_for_frames()

            # Get depth frame
            depth_frame = frames.get_depth_frame()

            # Colorize depth frame to jet colormap
            depth_color_frame = colorizer.colorize(depth_frame)

            # Convert depth_frame to numpy array to render image in opencv
            depth_color_image = np.asanyarray(depth_color_frame.get_data())

            black = np.zeros(np.shape(depth_color_image))
            # Render image in opencv window
            if i < frame_num:
                cv2.imshow(cow_num, depth_color_image)
                key = cv2.waitKey(1)
                i += 1
                # if pressed escape exit program
            elif i >= frame_num:
                cv2.imshow(cow_num, black)
                key = cv2.waitKey(1)

            if key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        pass
    print(f"COW: {cow_num}\n")
    video_score = input("Enter score : \n")
    print(f"saving ...{video_score}\n")
    video_data.loc[f, "file"] = file
    video_data.loc[f, "frame_num"] = frame_num
    video_data.loc[f, "score"] = video_score

video_data.to_csv("video_assesment.csv", index = False)