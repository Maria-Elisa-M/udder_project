import numpy as np
from scipy import ndimage as ndi
import shapely
import math
import skimage as ski
import os
from skimage.morphology import dilation, square
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.restoration import inpaint


def area_ratio(labels):
    values = np.max(labels)
    areas = []
    for value in range(values):
        area = len(labels[labels==value+1])
        areas.append(area)
    return max(areas)/min(areas)

def get_angle(right_kp, left_kp):
    angle = np.arctan2(right_kp[1]-left_kp[1], right_kp[0]-left_kp[0])
    return angle
def get_center(right_kp, left_kp):
    return shapely.centroid(MultiPoint([right_kp, left_kp])).coords[0] 
def get_orientation(right_kp, left_kp):
    if right_kp[0] < left_kp[0]:
        orientation = -1 # up 
    else: 
        orientation = 1 # down
    return orientation

def sep_points(right_kp, left_kp, udder_shp, box, limit = 10):
    global im_width, im_height
    wdist = np.linalg.norm(right_kp-left_kp)
    cnt = 0
    k = get_orientation(right_kp,left_kp)
    while (wdist < min(box[1, 0],  box[1, 1])/2) & (cnt < limit):
        angle = get_angle(right_kp, left_kp)
        nrb_point = [right_kp[0] + 10*np.cos(-k*angle), right_kp[1] + 10*np.sin(-k*angle)]
        nlb_point = [left_kp[0] - 10*np.cos(-k*angle), left_kp[1] - 10*np.sin(-k*angle)]
        # make sure they are still inside the udder
        if (udder_shp.contains(shapely.Point(nrb_point))): # & (nrb_point[0]>0) & (nrb_point[0] <= im_width) & (nrb_point[1]>0) & (nrb_point[1] <= im_height):
            # update points 
            right_kp = np.array(nrb_point)
        if (udder_shp.contains(shapely.Point(nlb_point))):# & (nlb_point[0]>0) & (nlb_point[0] <= im_width) & (nlb_point[1]>0) & (nlb_point[1] <= im_height):
            left_kp = np.array(nlb_point)
        wdist = np.linalg.norm(right_kp-left_kp)
        cnt += 1
    return (np.floor(right_kp).astype(int), np.floor(left_kp).astype(int))

class udder_object:
    def __init__(self, file, img_dir, label_dir, array = 0):
        if img_dir != "":
            cow = file.split("_")[0]
            udder = ski.io.imread(os.path.join(img_dir, cow, file))
        else:
            udder = array
        self.img = udder
        self.label = file.replace(".tif", ".txt")
        self.size = udder.shape
        self.sg_dir = os.path.join(label_dir, "segments")
        self.kp_dir =  os.path.join(label_dir, "keypoints")
        
    def get_segment(self):
        with open(os.path.join(self.sg_dir, self.label), "r") as f:
            mask = np.array([float(point) for point in f.read().split(" ")][1:])
        return mask.reshape((int(len(mask)//2),2))
    
    def get_keypoints(self):
        # keypoints are in x,y oder
        with open(os.path.join(self.kp_dir, self.label), "r") as f:
            data =  [float(point) for point in f.read().split(" ")]
            points = np.array(data[5:])
        points = points.reshape((4,3))
        points[:, 0] = points[:, 0] * self.size[1]
        points[:, 1] = points[:, 1] * self.size[0]
        return points
    
    def get_box(self):
        with open(os.path.join(self.kp_dir, self.label), "r") as f:
            data =  [float(point) for point in f.read().split(" ")]
            box = np.array(data[1:5])
        box = box.reshape((2,2))
        box[:, 0] = box[:, 0] * self.size[1]
        box[:, 1] = box[:, 1] * self.size[0]
        box[0, 0] = box[0, 0] - box[1, 0]/2
        box[0, 1] = box[0, 1] - box[1, 1]/2
        return box
    
    def get_mask(self):
        polygon = [[coord[1] * self.size[0], coord[0]* self.size[1]] for coord in  self.get_segment()]
        return ski.draw.polygon2mask(self.size, polygon)
    
    def get_shape(self):
        polygon2 = [[coord[0]* self.size[1], coord[1] * self.size[0]] for coord in self.get_segment()]
        return shapely.Polygon(polygon2)
    
def watershed_labels(points2, udder, dil_factor = 30, ratio_limit = 4, iter_limit = 10):
    udder_mask = udder.get_mask()
    masked_udder = udder.img*udder_mask
    mask1 = np.zeros(udder.size)
    # marker locations
    mask1[points2[0, 1], points2[0,0]] = True
    mask1[points2[1, 1], points2[1,0]] = True
    mask1[points2[2, 1], points2[2,0]] = True
    mask1[points2[3, 1], points2[3,0]] = True

    mask1 = dilation(mask1,  square(dil_factor))
    markers, _ = ndi.label(mask1)
    # find segments
    labels = watershed(masked_udder, markers = markers, mask = udder_mask, watershed_line=True)
    # area of labels
    ratio = area_ratio(labels)
    cnt = 0
    # print(f"{cnt} cow: {cow}, ratio: {ratio}")
    while (ratio > ratio_limit) & (cnt < iter_limit): # and the number of segements is 4
        mask1 = dilation(mask1,  square(10))
        markers, _ = ndi.label(mask1)
        labels2 = watershed(masked_udder, markers = markers, mask = udder_mask, watershed_line=True)
        num_segments = np.max(labels2)
        if num_segments < 4:
            break
        else:
            labels = labels2
        ratio = area_ratio(labels)
        num_segments = np.max(labels2)
        cnt+= 1
    return labels