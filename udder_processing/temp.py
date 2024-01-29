import os
import watershed_udder as wu
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyrealsense2 as rs
from astropy.convolution import Gaussian2DKernel, convolve,interpolate_replace_nans
import open3d as o3d

from scipy.linalg import lstsq
from scipy.spatial import Delaunay
import math
import functools 

def points_toworld(points):
    points2 = points.copy()
    for i in range(len(points)):
        points2[i, :] = rs.rs2_deproject_pixel_to_point(intr, [points[i, 0], points[i, 1]], points[i, 2])
    return points2


# list files 
dirpath = os.getcwd()
ws_dir = r"validate_watershed\watershed_segments"
corr_dir = r"validate_watershed\watershed_correspondence"
label_dir = os.path.join(dirpath, r"validate_watershed\pred_labels")
kp_dir = os.path.join(label_dir, r"keypoints")
sg_dir = os.path.join(label_dir, r"segments")
img_dir = os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_video\depth_images")
filenames = [file.replace(".npy", "") for file in os.listdir(ws_dir)]

video_path =  os.path.join(os.path.normpath(dirpath + os.sep + os.pardir), r"udder_video\video_files\example_video.bag")

config = rs.config()
rs.config.enable_device_from_file(config, video_path, repeat_playback = False)
pipeline = rs.pipeline()
cfg = pipeline.start(config) # Start pipeline and get the configuration it found
profile = cfg.get_stream(rs.stream.depth) # Fetch stream profile for depth stream
intr = profile.as_video_stream_profile().get_intrinsics() # Downcast to video_stream_profile and fetch intrinsics

for file in filenames[:1]:
    # udder object
    udder = wu.udder_object(file + ".tif", img_dir, label_dir, array = 0)
    # read image
    img = udder.img
    # read labels
    segment = udder.get_segment()
    points = udder.get_keypoints()
    # reas WS segmentation
    ws_label = np.load(os.path.join(ws_dir, file + ".npy"))
    kp_ws = pd.read_csv(os.path.join(corr_dir, file +".csv")).loc[0].to_dict()
    new_kp = wu.update_kp(kp_ws, ws_label, img)
    plt.imshow(img*udder.get_mask())
    plt.plot(new_kp["lf"][0], new_kp["lf"][1], "*r")
    plt.plot(new_kp["rf"][0], new_kp["rf"][1], "*b")
    plt.plot(new_kp["lb"][0], new_kp["lb"][1], "*r")
    plt.plot(new_kp["rb"][0], new_kp["rb"][1], "*b")
    plt.show()
    

#%%

scale = 0.0001
img = udder.img.copy().astype(float)
img[img ==0] = np.nan
kernel = Gaussian2DKernel(x_stddev=1)
udder_conv = convolve(img, kernel)
udder_conv[np.isnan(udder_conv)] = 0

masked_udder =  udder.get_mask() * udder_conv 
rows, cols = np.nonzero(masked_udder)
values = masked_udder[rows, cols]
udder_points = np.column_stack((np.transpose(cols), np.transpose(rows), np.transpose(values))).astype(float)
udder_points[:, 2] = udder_points[:, 2] *scale
pts = points_toworld(udder_points)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pts)

pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#%%

segment = np.round([[coord[1] * udder.size[0], coord[0]* udder.size[1]] for coord in udder.get_segment()]).astype(int)
cols = segment[:, 1]
rows = segment[:, 0]
values = udder_conv[rows, cols]*scale
segment_points = np.column_stack((np.transpose(cols), np.transpose(rows), np.transpose(values))).astype(float)
sgpts = points_toworld(segment_points)


X = np.column_stack((np.ones((len(sgpts), 1)), sgpts[:, :2]))
z = np.transpose(sgpts[:, 2])

b = np.matrix(z).T
A = np.matrix(X)

fit, residual, rnk, s = lstsq(A, b)

predz =  fit[1] * pts[:,0] + fit[2] * pts[:,1] + fit[0]
croped = pts.copy()
croped = croped[croped[:,2] <= predz[:]]

pcd.points = o3d.utility.Vector3dVector(croped)
pcd.estimate_normals(
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))


a = fit[1][0]
b = fit[2][0]
c = -1
d = fit[0][0]
cos_theta = c / math.sqrt(a**2 + b**2 + c**2)
sin_theta = math.sqrt((a**2+b**2)/(a**2 + b**2 + c**2))
u_1 = b / math.sqrt(a**2 + b**2 )
u_2 = -a / math.sqrt(a**2 + b**2)
rotation_matrix = np.array([[cos_theta + u_1**2 * (1-cos_theta), u_1*u_2*(1-cos_theta), u_2*sin_theta],
                            [u_1*u_2*(1-cos_theta), cos_theta + u_2**2*(1- cos_theta), -u_1*sin_theta],
                            [-u_2*sin_theta, u_1*sin_theta, cos_theta]])
pcd.rotate(rotation_matrix)

downpdc = pcd.voxel_down_sample(voxel_size=0.0001)
xyz = np.asarray(downpdc.points)
xy_catalog = []
for point in xyz:
    xy_catalog.append([point[0], point[1]])
tri = Delaunay(np.array(xy_catalog))

surface = o3d.geometry.TriangleMesh()
surface.vertices = o3d.utility.Vector3dVector(xyz)
surface.triangles = o3d.utility.Vector3iVector(tri.simplices)

o3d.visualization.draw_geometries([surface], mesh_show_wireframe=True)

#%%
def get_triangles_vertices(triangles, vertices):
    triangles_vertices = []
    for triangle in triangles:
        new_triangles_vertices = [vertices[triangle[0]], vertices[triangle[1]], vertices[triangle[2]]]
        triangles_vertices.append(new_triangles_vertices)
    return np.array(triangles_vertices)

def volume_under_triangle(triangle):
    p1, p2, p3 = triangle
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    x3, y3, z3 = p3
    return abs((z1+z2+z3)*(x1*y2-x2*y1+x2*y3-x3*y2+x3*y1-x1*y3)/6)


volume = functools.reduce(lambda a, b:  a + volume_under_triangle(b), get_triangles_vertices(surface.triangles, surface.vertices), 0)


print(f"The volume of the stockpile is: {round(volume*1000, 4)} m3")