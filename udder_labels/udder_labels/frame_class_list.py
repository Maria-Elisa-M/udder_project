import os
import pandas as pd

# cow, file name, class
# list all labels
label_path = os.path.join(os.getcwd(), r'labels\class') 
label_list = os.listdir(label_path)
filenames = [file.replace(".txt", "") for file in label_list]
cow_list = [file.split("_")[0] for file in label_list]
class_list = []
for label in label_list:
    with open(os.path.join(label_path, label), "r") as f:
        class_list.append(f.read())
        
df = pd.DataFrame({"filename":filenames , "cow":cow_list, "frame_class": class_list})
df.to_csv("frame_class_list.csv", index = False)