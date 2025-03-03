# Udder pipeline 
Description <br/>
Goal: <br/>
This folder has the last running version of the scripts.  <br/>
The working versions are under the different folders of this repository. <br/>

# Diagram
<img src = "diagram\udder_flowchart.png" width = 600>

# Input
Description <br/>

# Output
Description <br/>
Temp output <br/>
Labels, pointclouds, images, etc.
Final output <br/>
Feature dictionaries, feature tables

# Directory structure

- udder_pipeline
  - Readme
  - models
    - frame_classify
    - udder_segment

 - scripts
    - udder_modules
    - 01_get_deptharrays.py
    - 02_predict_labels.py
    - 03_watershed_segment.py
    - 04_predict_class_ws.py
    - 05_write_good_frames.py
    - 06_features_shape.py
    - 07_ls_pointclouds.py
    - 08_features_volumes.py
    - 09_features_teat_v1.py
    - 10_feature_table.py

# How to run

## config file

This is a json file specifyng the following paths:<br/>
  * model_path
  * temp_path
  * output_path


## environment