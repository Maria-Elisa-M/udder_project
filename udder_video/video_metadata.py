import pandas as pd
import os
import datetime

# info
path = os.getcwd()
dirpath = os.path.join(path, "video_files")
outpath = os.path.join(path, "video_metadata")

# videos 1: Marias computer -vms 3 and 4 right side
# videos 2: Ghuilhermes computer -vms 1 and 2 left side mostly*
# videos 3: Labs computer -vms 1 and 2 left side
computer_dict = {"videos_1": "maria", "videos_2":"guilherme", "videos_3": "lab"}
robotside_dict = {"videos_1": "right", "videos_2": "left", "videos_3": "left"}
timedelta_dict = {"videos_1": 2, "videos_2": 0, "videos_3": 2}
farmname ="laufenberg"
out_filename = "video_metadata_20231117.csv"

# build df
dir_list = os.listdir(dirpath)
file_dict = {}
for folder in dir_list:
    file_dict[folder] = os.listdir(os.path.join(dirpath, folder))

columnames =  ["cow", "farmname","robotside", "filename","directory", "date","time" ,"size", "computer"]
all_files = pd.DataFrame(columns =columnames)

for folder in dir_list:
    folder_path = os.path.join(dirpath, folder)
    file_list = os.listdir(folder_path)
    filedf = pd.DataFrame(columns = columnames, index = range(len(file_list)))
    size = [os.path.getsize(os.path.join(folder_path, file)) for file in file_list]
    date_times = [datetime.datetime.fromtimestamp(os.path.getctime(os.path.join(folder_path, file))) + datetime.timedelta(hours =timedelta_dict[folder]) for file in file_list]
    filedf["cow"] = [file.split("_")[0] for file in file_list]
    filedf["filename"] = file_list
    filedf["farmname"] = farmname
    filedf["robotside"] = robotside_dict[folder]
    filedf["computer"] = computer_dict[folder]
    filedf["directory"] = folder
    filedf["date"] = [time.strftime('%Y%m%d') for time in date_times]
    filedf["time"] = [time.strftime('%H:%M:%S') for time in date_times]
    filedf["size"] = size
    
    all_files = pd.concat([all_files, filedf], axis=0, ignore_index = True)

#save file
all_files.to_csv(os.path.join(outpath, out_filename), index = False)