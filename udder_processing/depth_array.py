import os
import pyrealsense2 as rs
import rosbag
import numpy as np
from tifffile import imwrite

dirpath = os.getcwd()
# video path 
video_path = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r'udder_video')
with open(os.path.join(video_path, r"filelist_topred.txt"), "r") as f:
    video_files = [file.replace("\n", "").split(",") for file in f.readlines()]
file_list = [os.path.join(video_path, "video_files", file[0], file[1]) for file in video_files]


# %%
def get_num_frames(filepath):
    topic = "/device_0/sensor_0/Depth_0/image/data"
    bag = rosbag.Bag(filepath, "r")
    nframes = int(bag.get_type_and_topic_info()[1][topic][1])
    return nframes

# %%
for filepath in file_list:
    nframes = get_num_frames(filepath)
    filename = filepath.split("\\")[-1]
    
    try:
        config = rs.config()
        rs.config.enable_device_from_file(config, filepath, repeat_playback = False)
        pipeline = rs.pipeline()
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        profile = pipeline.start(config)
        playback = profile.get_device().as_playback()
        playback.set_real_time(False)
        i = 0
        while True:
            frames = pipeline.wait_for_frames()
            playback.pause()
            depth_frame = frames.get_depth_frame()
            if i == 0:
                depth_array = np.empty((nframes, np.array(depth_frame.get_data()).shape[0], np.array(depth_frame.get_data()).shape[1]), dtype= "uint16")
            depth_array[i] = np.expand_dims(np.array(depth_frame.get_data()), axis=0)   
            i += 1
            playback.resume()
    except RuntimeError:
        cow = str(int(filename.split("_")[0]))
        video = "_".join(filename.replace(".bag", "").split("_")[1:3])  
        fname_depth =  cow + "_" + video + "_depthframe_" + str(i)
        np.save(os.path.join("arrays", fname_depth), depth_array[0:i])
        
    finally:
        pipeline.stop()
    
    print(filename)