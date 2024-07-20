import numpy as np
import os
import watershed_udder as wu
import pandas as pd

dirpath = os.getcwd()
label_dir = os.path.join(dirpath, r"pred_labels")
kp_dir = os.path.join(label_dir, r"keypoints")
sg_dir = os.path.join(label_dir, r"segments")
im_dir = ""
out_dir = os.path.join(label_dir, r"watershed_segments")
out_dir2 = os.path.join(label_dir, r"watershed_correspondence")
# list of files
inpath = "frames_tosave"
array_path = "arrays"
file_list = os.listdir(array_path)

# for file in list  read content
for file in file_list[45:]:
    cow = file.split("_")[0]
    src = os.path.join(array_path, file)
    depth_array = np.load(src, mmap_mode="r")
    good_frames = "_".join(file.split("_")[:3]) +".txt"
    with open(os.path.join(inpath, good_frames), "r") as f:
        frames = f.read()
        if frames != "":
            frames = [int(num) for num in frames.split(",")]
            print(f"\n{cow}: {len(frames)}")
            cnt = 1
            for frame in frames:
                file = "_".join(file.split("_")[:3]) +"_frame_" + str(frame) + ".tif"
                img = depth_array[frame]
                out_name = file.replace(".tif", ".npy")
                udder = wu.udder_object(file,im_dir, label_dir, img)
                udder_shp = udder.get_shape()
                udder_box = udder.get_box()
                points = udder.get_keypoints()
                udder_box = udder.get_keypoints()
                udder_mask = udder.get_mask()
                masked_udder = udder.img*udder_mask
                mask1 = np.zeros(udder.size)
                points2 =np.round(points,0).astype(int)

                lf_kp = points[0, :2]
                rf_kp = points[1, :2]
                lb_kp = points[2, :2]
                rb_kp = points[3, :2]

                new_front = wu.sep_points(rf_kp, lf_kp, udder_shp, udder_box)
                points2[0, :2] = new_front[0]
                points2[1, :2] = new_front[1]

                new_back = wu.sep_points(rb_kp, lb_kp, udder_shp, udder_box)
                points2[2, :2] = new_back[0]
                points2[3, :2] = new_back[1]

                labels = wu.watershed_labels(points2, udder)
                np.save(os.path.join(out_dir, out_name), labels)
                
                temp = pd.DataFrame(wu.find_correspondence(points2, labels), index = [0])
                temp.to_csv(os.path.join(out_dir2, file.replace(".tif", ".csv")), index = False)
                
                print(f"{cnt}: {file}")
                cnt +=1