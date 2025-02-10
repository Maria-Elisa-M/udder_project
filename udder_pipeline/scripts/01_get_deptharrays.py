# Maria Elisa Montes
# Working version: get_frames
# last update: 2025-02-10

import os
import sys
import json
import pyrealsense2 as rs
import numpy as np
import rosbag
from tifffile import imwrite

def mk_dir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        
def get_num_frames(filepath):
    topic = "/device_0/sensor_0/Depth_0/image/data"
    bag = rosbag.Bag(filepath, "r")
    nframes = int(bag.get_type_and_topic_info()[1][topic][1])
    return nframes

def get_depth_frame(filepath, filename, outpath):
    nframes = get_num_frames(filepath) 
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
        outname = filename.replace(".bag", ".npy")
        np.save(os.path.join(outpath, outname), depth_array[0:i])
    finally:
        pipeline.stop()
        
dirpath = os.getcwd()
config_path = os.path.join(dirpath, "udder_config.json")

# Open and read the JSON file
with open(config_path, 'r') as file:
    data = json.load(file)
input_dir = data["input_path"]
output_dir = data["temp_path"]

inpath =  os.path.join(input_dir, "videos")
outpath = os.path.join(output_dir, "arrays")
mk_dir(outpath)


for file in os.listdir(inpath):
    filepath = os.path.join(inpath, file)
    print(file)
    get_depth_frame(filepath, file, outpath)
    print("---done\n")