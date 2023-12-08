import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

video = pd.concat([pd.read_csv("video_metadata_20231117_p2.csv"), pd.read_csv("video_metadata_20231117_p1.csv"), pd.read_csv("video_metadata_20231117_p3.csv")], axis = 0, ignore_index = True)
cow_typos = {732: 723, 1179: 1279, 1332:1432, 934:1274}
video.replace({"cow": cow_typos}, inplace = True)

# VMS reoports are semicolon separates and column names have spacces and capital letters
cows_df = pd.read_csv("animal_info.csv", sep = ";")
cows_df.columns = [col.replace(" ", "_").lower() for col in cows_df.columns]
milk_df = pd.read_csv("milk_quarter3.csv", sep = ";") 
milk_df.columns = [col.replace(" ", "_").lower() for col in milk_df.columns]

# fix time and date to make them match
# I give a +-10 min margin form video to milking
video["start_hour0"]=[datetime.strptime(date, '%H:%M:%S').hour for date in video.time]
video["start_hour1"]=[datetime.strptime(date, '%H:%M:%S') + timedelta(minutes = 10) for date in video.time]
video["start_hour1"]=[hour.hour for hour in video.start_hour1]
video["start_hour2"]=[datetime.strptime(date, '%H:%M:%S') - timedelta(minutes = 10) for date in video.time]
video["start_hour2"]=[hour.hour for hour in video.start_hour2]

# milk report time is in AM -PM format
milk_df["start_hour"]=[datetime.strptime(str(datetime.strptime(date, '%m/%d/%Y %H:%M %p')), '%Y-%m-%d %H:%M:%S').hour for date in milk_df.begin_time]
milk_df["day"]=[datetime.strptime(str(datetime.strptime(date, '%m/%d/%Y %H:%M %p')), '%Y-%m-%d %H:%M:%S').day for date in milk_df.begin_time]
milk_df["AMPM"] = [re.findall(r"[A-Z]+", date)[0] for date in milk_df.begin_time]
for i in range(len(milk_df["start_hour"])):
    if (milk_df.loc[i, "AMPM"] == "PM") & (milk_df.loc[i, "start_hour"] < 12):
        milk_df.loc[i, "start_hour"]  =  milk_df.loc[i, "start_hour"] + 12
    elif (milk_df.loc[i, "AMPM"] == "AM") & (milk_df.loc[i, "start_hour"] ==12):
        milk_df.loc[i, "start_hour"]  =  0
        
# verify that all  milkins are from november 17 
milk_df["end_hour"]=milk_df["start_hour"]+1
milk_df = milk_df[milk_df.day ==17]
milk_df = milk_df.drop(["date","day"], axis = 1)