# -*- coding: utf-8 -*-
"""
Created on Thu Oct  5 07:58:25 2023

@author: Maria
"""

import os
import pyrealsense2 as rs
import pandas as pd
#%%

def listDir(path):
    listdir = []
    for d in os.listdir(path):
        if os.path.isdir(os.path.join(path, d)):
            listdir.append(d)
    return listdir


def listFiles(path, label):
    listfiles = []
    for file in os.listdir(path):
        listfiles.append(file)
    return listfiles
      
#%%
path = os.getcwd()
inputpath= path + "\\3Dvideos-renamed"    
listfiles = listFiles(inputpath, ".bag")

#%%
data_intr = pd.DataFrame(index = range(len(listfiles)), columns = ['filename', 'intinsics'])
for i, file in enumerate(listfiles):
    config = rs.config()
    rs.config.enable_device_from_file(config, os.path.join(inputpath, file), repeat_playback = False)
    pipeline = rs.pipeline()
    cfg = pipeline.start(config) # Start pipeline and get the configuration it found
    profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
    intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics
    data_intr.loc[i,"filename"] = file
    data_intr.loc[i,'intinsics'] = repr(intr)
    
#%%
data_intr .to_csv("round2_intrinsics.csv", index=False)