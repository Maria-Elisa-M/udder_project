# Udder pipeline 
Description
Goal: 
This folder has the last running version of the scripts. 
The working versions are under the different folders of this repository.

# Diagram
<img src = "diagram\udder_flowchart.png" width = 600>

# Input
Description

# Output
Description

# Directory structure

udder_pipeline
  Readme
  models
    frame_classify
    udder_segment

  scripts
    udder_modulees
    01_get_deptharrays.py
    02_predict_labels.py
    03_watershed_segment.py
    04_predict_class_ws.py
    05_write_good_frames.py
    06_describe_shape.py

# How to run

## config file

This is a json file specifyng the following paths:
*model_path
*input_path
*temp_path
*output_path


## envronment