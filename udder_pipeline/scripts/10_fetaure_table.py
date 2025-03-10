import numpy as np
import json 
import pandas as pd
import os


dirpath = os.getcwd()
feature_path = os.path.join(dirpath, "features_dict")
volume_path = os.path.join(feature_path, "volumes")
angle_path = os.path.join(feature_path, "angles")
distance_path = os.path.join(feature_path, "distance")
shape_path = os.path.join(feature_path, "shape")
teat_path = os.path.join(feature_path, "teat_length")

filenames = [file.replace(".json", "") for file in os.listdir(distance_path)]

sides = ['front', 'back', 'right', 'left']
teats = ["lf", "rf", "lb", "rb"]
quarters = ["udder"] + teats
shapes = ['peri', 'area', 'circ', 'exc']
quarters_shapes =  [q +'_'+ s for q in quarters for s in shapes]

eudist_df = pd.DataFrame(columns = sides, index = filenames)
gddist_df = pd.DataFrame(columns = sides, index = filenames)
angles_df = pd.DataFrame(columns = teats, index = filenames)
area_df = pd.DataFrame(columns =["lfrb", "rflb"], index = filenames)
volume_df = pd.DataFrame(columns = quarters, index = filenames)
sarea_df = pd.DataFrame(columns =  quarters, index = filenames)
shape_df = pd.DataFrame(columns =  quarters_shapes, index = filenames)
teat_df = pd.DataFrame(columns =  teats, index = filenames)

for file in filenames: 
    with open(os.path.join(distance_path, file + ".json")) as f:
        distance_dict = json.load(f)
    with open(os.path.join(volume_path, file + ".json")) as f:
        volume_dict = json.load(f)
    with open(os.path.join(angle_path, file + ".json")) as f:
        angle_dict = json.load(f)
    with open(os.path.join(shape_path, file + ".json")) as f:
        shape_dict = json.load(f)
    with open(os.path.join(teat_path, file + ".json")) as f:
        teat_dict = json.load(f)
        
    volume_df.loc[file, "udder"] = volume_dict["udder"]["volume"]
    sarea_df.loc[file, "udder"] = volume_dict["udder"]["sarea"]*100*100
    for shape in shape_dict["udder"].keys():
        col  = 'udder' +'_'+ shape
        shape_df.loc[file, col] = shape_dict['udder'][shape]
    for teat in angle_dict.keys():
        angles_df.loc[file, teat] = angle_dict[teat]["angle"]
    for teat in teat_dict.keys():
        teat_df.loc[file, teat] = teat_dict[teat]["length"]*1000
    for side in distance_dict.keys():
        eudist_df.loc[file, side] = distance_dict[side]["eu"]*100
        gddist_df.loc[file, side] = distance_dict[side]["geo"]*100
    for quarter in volume_dict['quarters'].keys():
        volume_df.loc[file, quarter] = volume_dict['quarters'][quarter]["volume"]
        sarea_df.loc[file, quarter] = volume_dict['quarters'][quarter]["sarea"] *100*100
    for quarter in shape_dict['quarters'].keys():
        qshape_dict = shape_dict['quarters'][quarter]
        for shape in qshape_dict.keys():
            col  = quarter+'_'+shape
            shape_df.loc[file, col] = qshape_dict[shape]
        
    area_df.loc[file, "lfrb"] = (angle_dict["lf"]["area"] + angle_dict["rb"]["area"]) *100*100
    area_df.loc[file, "rflb"] = (angle_dict["rf"]["area"] + angle_dict["lb"]["area"]) *100*100

volume_df.columns = [col +"_vol" for col in  volume_df.columns]
sarea_df.columns = [col +"_sarea" for col in  sarea_df.columns]
angles_df.columns = [col +"_angle" for col in  angles_df.columns]
teat_df.columns = [col +"_len" for col in  teat_df.columns]
eudist_df.columns = [col +"_eu" for col in  eudist_df.columns]
gddist_df.columns = [col +"_gd" for col in  gddist_df.columns]

merged_df = volume_df.join(sarea_df).join(angles_df).join(eudist_df).join(gddist_df).join(shape_df).join(teat_df).reset_index().rename(columns={'index': 'filename'})
cols = ['cow', 'frame'] +  merged_df.columns.tolist()
merged_df[["cow", "frame"]] = [[file.split("_")[0], file.split("_")[-1]] for file in merged_df.filename]
merged_df = merged_df.loc[:, cols]
merged_df.to_csv(os.path.join(feature_path, "feature_table.csv"), index = False)