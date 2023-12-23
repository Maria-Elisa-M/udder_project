import os
import sys
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

def get_depth_frame(filepath, filename, outpath, outpath2, outpath3):
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
        colorizer = rs.colorizer()
        while True:
            frames = pipeline.wait_for_frames()
            playback.pause()
            depth_frame = frames.get_depth_frame()
            depth_color_frame = colorizer.colorize(depth_frame)
            if i == 0:
                color_array = np.empty((nframes, np.array(depth_color_frame.get_data()).shape[0], np.array(depth_color_frame.get_data()).shape[1], 3), dtype= "uint8")
                depth_array = np.empty((nframes, np.array(depth_frame.get_data()).shape[0], np.array(depth_frame.get_data()).shape[1]), dtype= "uint16")
                
            color_array[i] = np.expand_dims(np.array(depth_color_frame.get_data()), axis=0)
            depth_array[i] = np.expand_dims(np.array(depth_frame.get_data()), axis=0)   
            i += 1
            playback.resume()
            
    except RuntimeError:
        cow = str(int(filename.split("_")[0]))
        video = "_".join(filename.replace(".bag", "").split("_")[1:3])
        arraypath = os.path.join(outpath1, cow)
        colorpath = os.path.join(outpath2, cow)
        depthpath = os.path.join(outpath3, cow)
        mk_dir(arraypath)
        mk_dir(colorpath)
        mk_dir(depthpath)
        fname_color = cow + "_" + video + "_colorframe_" + str(i)
        fname_depth =  cow + "_" + video + "_depthframe_" + str(i)
        np.save(os.path.join(arraypath, fname_depth), depth_array[0:i])
        np.save(os.path.join(arraypath, fname_color), color_array[0:i])
        for j in range(0, nframes):
            imwrite(os.path.join(colorpath, cow + "_" + video + "_frame_"+str(j+1)+".tif"), color_array[j])
            imwrite(os.path.join(depthpath, cow + "_" + video + "_frame_"+str(j+1)+".tif"), depth_array[j])

    finally:
        pipeline.stop()
        
path = os.getcwd()
inpath =  os.path.join(path, "video_files")
outpath1 = os.path.join(path, "arrays")
outpath2 = os.path.join(path, "color_images")
outpath3 = os.path.join(path, "depth_images")

with open("filelist_toframe.txt", "r") as f:
    video_files = [file.replace("\n", "").split(",") for file in f.readlines()]

for file in video_files:
    filepath = os.path.join(inpath, file[0], file[1])
    print(file[1])
    get_depth_frame(filepath, file[1], outpath1, outpath2, outpath3)
    print("---done\n")