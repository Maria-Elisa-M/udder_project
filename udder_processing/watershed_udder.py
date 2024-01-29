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
from skimage.measure import find_contours
from astropy.convolution import Gaussian2DKernel, convolve,interpolate_replace_nans
from shapely import LineString, MultiPoint, Polygon
from skimage.transform import rotate

def area_ratio(labels):
    values = np.unique(labels)
    values = values[values!=0]
    areas = []
    for value in values:
        area = len(labels[labels==value])
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
        if (udder_shp.contains(shapely.Point(nrb_point))):
            # update points 
            right_kp = np.array(nrb_point)
        if (udder_shp.contains(shapely.Point(nlb_point))):
            left_kp = np.array(nlb_point)
        wdist = np.linalg.norm(right_kp-left_kp)
        cnt += 1
    return (np.floor(left_kp).astype(int), np.floor(right_kp).astype(int))

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
    miss_mask = udder.img.copy()
    miss_mask[: :] = 0
    miss_mask[udder == 0] = 1
    inp_udder = inpaint.inpaint_biharmonic(udder.img, miss_mask)
        
    udder_mask = udder.get_mask()
    masked_udder = inp_udder*udder_mask
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

def find_correspondence(points, labels):
    lf_point = shapely.Point(points[0, :2])
    rf_point = shapely.Point(points[1, :2])
    lb_point = shapely.Point(points[2, :2])
    rb_point = shapely.Point(points[3, :2])

    point_dict = {"lf": lf_point, "rf": rf_point, "lb": lb_point, "rb": rb_point}
    quarter_dict ={}

    values = np.unique(labels)
    values = values[values!=0]
    for value in values:
        labels2 = labels.copy()
        labels2[labels2 != value] =0
        labels2[labels2 > 0] = 1
        contour = find_contours(labels2)[0]
        polygon = [[coord[1], coord[0]] for coord in contour]
        quarter_shp = shapely.Polygon(polygon)
        quarter_dict[value] = quarter_shp
        
    correspondence_dict = {}
    for name in point_dict.keys():
        pt = point_dict[name]
        for key, quarter in quarter_dict.items():
            if quarter.contains(pt):
                correspondence_dict[name] = key
    
    return correspondence_dict

# udder lines functions

# get the angle btween kp
def get_angle(right_kp, left_kp):
    angle = np.arctan2(right_kp[1]-left_kp[1], right_kp[0]-left_kp[0])
    return angle
# get the center between kp
def get_center(right_kp, left_kp):
    return shapely.centroid(MultiPoint([right_kp, left_kp])).coords[0] 
# udder orienttion in image
def get_orientation(right_kp, left_kp):
    if right_kp[0] < left_kp[0]:
        orientation = -1 # up 
    else: 
        orientation = 1 # down
    return orientation
# rotate the udder so that kp are side by side
def rotate_udder(udder, right_kp, left_kp):
    k = get_orientation(right_kp, left_kp)
    center = get_center(right_kp, left_kp)
    angle = get_angle(right_kp, left_kp)
    rotated_udder = rotate(udder, np.rad2deg(k*angle), center = center, preserve_range = True)
    return rotated_udder
# rotate the kp 
def rotate_points(right_kp, left_kp):
    k = get_orientation(right_kp, left_kp)
    points = np.concatenate([[right_kp], [left_kp]])
    points2 = points.copy()
    angle = get_angle(right_kp, left_kp)
    center = get_center(right_kp, left_kp)
    rot_mat = np.array([[np.cos(-k*angle), -np.sin(-k*angle)], [np.sin(-k*angle), np.cos(-k*angle)]])
    #
    points2[:, 0] = points[:, 0] - center[0]
    points2[:, 1] = points[:, 1] - center[1]
    # 
    points2 = np.transpose(np.dot(rot_mat, np.transpose(points2[:, :2])))
    points2[:, 0] = points2[:, 0] + center[0]
    points2[:, 1] = points2[:, 1] + center[1]
    rotated_points = points2.copy()
    
    return rotated_points
# get the depth values from one kp to the other
def udder_line(udder_object, udder_shp, rf_kp, lf_kp):
    img = udder_object.img.copy().astype(float)
    im_width =udder_object.size[1]
    img[img ==0] = np.nan
    kernel = Gaussian2DKernel(x_stddev=1)
    udder_conv = convolve(img, kernel)
    udder2 = rotate_udder(udder_conv, rf_kp, lf_kp)
    points2 = rotate_points(rf_kp, lf_kp)
    yloc = np.floor(points2[0,1]).astype(int)
    # fig, ax = plt.subplots()
    # for i in range(0,1):
    yloc2 = yloc #  + i 
    line = LineString([(0, yloc2), (im_width, yloc2)])
    intersection = udder_shp.exterior.intersection(line).geoms
    endpoints = np.array([list(intersection[0].coords[0]), list(intersection[1].coords[0])])
    start = np.floor(endpoints[np.argmin(endpoints[:, 0])]).astype(int)
    end = np.floor(endpoints[np.argmax(endpoints[:, 0])]).astype(int)
    line_vals = udder2[yloc2][list(range(start[0], end[0]))]
    x = np.array(list(range(start[0],  end[0])))
    y = np.array([yloc]*len(x))
    z = line_vals
    return np.column_stack((x, y, z))

# derotate points 
def derotate_points(right_kp, left_kp, rotated_points):
    k = get_orientation(right_kp, left_kp)
    angle = -get_angle(right_kp, left_kp)
    center = get_center(right_kp, left_kp)
    rot_mat = np.array([[np.cos(-k*angle), -np.sin(-k*angle)], [np.sin(-k*angle), np.cos(-k*angle)]])
    points = rotated_points.copy()
    points[:, 0] = rotated_points[:, 0] - center[0]
    points[:, 1] = rotated_points[:, 1] - center[1]
    
    points = np.transpose(np.dot(rot_mat, np.transpose(points[:, :2])))
    
    points[:, 0] = points[:, 0] + center[0]
    points[:, 1] = points[:, 1] + center[1]
    
    derotated_points = np.floor(points).astype(int)
    return derotated_points

def update_kp(kp_ws, ws_label, img):
    newkp_dict = {}
    # fig, axs = plt.subplots(ncols = 4, nrows= 1, figsize = (12, 4))
    for key in kp_ws.keys():
        label = kp_ws[key]
        mask = ws_label.copy()
        mask[mask!= label] = 0
        mask[mask == label] = 1
        quarter = (mask*img).astype(float)
        quarter[quarter==0] =np.nan
        mins = np.argwhere(quarter== np.nanmin(quarter))
        x = np.round(np.median(mins[:, 1]), 0).astype(int)
        y = np.round(np.median(mins[:, 0]), 0).astype(int)
        newkp_dict[key] = (x,y)
    return newkp_dict