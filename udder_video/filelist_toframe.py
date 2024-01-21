import pandas as pd
import os
import numpy as np

# info
path = os.getcwd()
dirpath = os.path.join(path, "video_metadata")
files_df = pd.read_csv(os.path.join(dirpath,"video_metadata_20231117.csv"))
print(f"There are {len(files_df)} videos")
# remove the files that give an error 
remove_files = ["1184_20231117_172549.bag", "1223_20231117_153008.bag", "738_20231117_104922.bag", "855_20231117_170701.bag"]
files_df = files_df[~files_df.filename.isin(remove_files)]
print(f"There are {len(files_df)} videos after removing those that give an error")

print(f"There are {len(np.unique(files_df.cow))} unique cows")
files_df["time2"] = pd.to_datetime(files_df["time"], format= '%H:%M:%S')
files_df = files_df.sort_values(by='time2').reset_index(drop= True)
# select 25 cows from each computer 
cow_list1 = np.concatenate((np.unique(files_df[files_df.computer == "guilherme"].cow)[:25], \
                              np.unique(files_df[files_df.computer == "maria"].cow)[:25]))

cow_list2 =np.concatenate((np.unique(files_df[files_df.computer == "guilherme"].cow)[25:], np.unique(files_df[files_df.computer == "maria"].cow)[25:], np.unique(files_df[files_df.computer == "lab"].cow)))

# keep only the first video for each coe in the list
file_list = []
for cow in cow_list1:
    cow_files = files_df[files_df.cow == cow].reset_index(drop= True)
    file_list.append( ",".join(cow_files.iloc[0][["directory", "filename"]]))

with open("filelist_toframe.txt", "w") as f:
    f.write("\n".join(file_list))
    
file_list = []
for cow in cow_list2:
    cow_files = files_df[files_df.cow == cow].reset_index(drop= True)
    file_list.append( ",".join(cow_files.iloc[0][["directory", "filename"]]))

with open("filelist_topred.txt", "w") as f:
    f.write("\n".join(file_list))