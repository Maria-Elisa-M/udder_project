# Models
* [Frame classification model](https://github.com/Maria-Elisa-M/udder_project/tree/main/udder_models#frame-classification-model)
* [Udder segmentation model](https://github.com/Maria-Elisa-M/udder_project/tree/main/udder_models#udder-segmentation-model)
* [Teat keypoint detection model](https://github.com/Maria-Elisa-M/udder_project/tree/main/udder_models#teat-keypoint-detection-model)

## Frame classification model
* Class 1 = good frame
* Class 0 = bad frame
See udder_labels for more details **add link**

* Yolo V8 = pt model
* settings
### Training, validation and test sets
* A: DCC first round 2021-06-25
* B: DCC second round 2021-10-22
* D: Robot farm Marias's computer 2023-11-17
* C: Robot farm Guilherme's computer 2023-11-17

<img src = "plots\collection_groups.png" width = 600>

#### Number of cows in each set
<table>
  <tr>
    <th>Set</th>
    <th>A</th>
    <th>B</th>
    <th>C</th>
    <th>D</th>
    <th>Grand Total</th>
  </tr>
  <tr>
    <td>Train</td>
    <td>17</td>
    <td>22</td>
    <td>15</td>
    <td>15</td>
    <td>69</td>
  </tr>
  <tr>
    <td>Val</td>
    <td>6</td>
    <td>6</td>
    <td>5</td>
    <td>5</td>
    <td>22</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>6</td>
    <td>6</td>
    <td>5</td>
    <td>5</td>
    <td>22</td>
  </tr>
  <tr>
    <td>Total</td>
    <td>29</td>
    <td>34</td>
    <td>25</td>
    <td>25</td>
    <td>113</td>
  </tr>
</table>

#### Number of images in each set
<table>
  <tr>
    <th rowspan="2"> Set</th>
    <th colspan = "5">Class 0</th>
    <th colspan = "5">Class 1</th>
    <th rowspan="2">Total</th>
  </tr>  
  <tr>
    <th>A</th>
    <th>B</th>
    <th>C</th>
    <th>D</th>
    <th>Total 0</th>
    <th>A</th>
    <th>B</th>
    <th>C</th>
    <th>D</th>
    <th>Total 1</th>
  </tr>  
  <tr>
    <td>Train</td>
    <td>539</td>
    <td>229</td>
    <td>1365</td>
    <td>1911</td>
    <td>4044</td>
    <td>1678</td>
    <td>2599</td>
    <td>3000</td>
    <td>2843</td>
    <td>10120</td>
    <td>14164</td>
  </tr>
  <tr>
    <td>Val</td>
    <td>152</td>
    <td>-</td>
    <td>480</td>
    <td>580</td>
    <td>1212</td>
    <td>643</td>
    <td>832</td>
    <td>1000</td>
    <td>1000</td>
    <td>3475</td>
    <td>4687</td>
  </tr>
   <tr>
    <td>Test</td>
    <td>200</td>
    <td>15</td>
    <td>501</td>
    <td>618</td>
    <td>1334</td>
    <td>652</td>
    <td>832</td>
    <td>960</td>
    <td>1000</td>
    <td>3444</td>
    <td>4778</td>
  </tr>
   <tr>
    <td>Total</td>
    <td>891</td>
    <td>244</td>
    <td>2346</td>
    <td>3109</td>
    <td>6590</td>
    <td>2973</td>
    <td>4263</td>
    <td>4960</td>
    <td>4843</td>
    <td>17039</td>
    <td>23629</td>
  </tr>
</table>

### Results
<table>
  <tr>
    <th rowspan = "2"> Threshold</th>
    <th rowspan = "2"> Class</th>
    <th colspan = "2"> Pred class</th>
  </tr>
  <tr>
    <th> 0 </th>
    <th> 1 </th>
  </tr>
  <tr>
    <td rowspan = "2"> argmax </td>
    <td> 0 </td>
    <td> 687 </td>
    <td> 647 </td>
  </tr>
  <tr>
    <td> 1 </td>
    <td> 50 </td>
    <td> 3394 </td>
  </tr>
  <tr>
    <td rowspan = "2"> conf 0.8 </td>
    <td> 0 </td>
    <td> 812 </td>
    <td> 522 </td>
  </tr>
  <tr>
    <td> 1 </td>
    <td> 159 </td>
    <td> 3285 </td>
  </tr>
  <tr>
    <td rowspan = "2"> conf 0.9 </td>
    <td> 0 </td>
    <td> 953 </td>
    <td> 381 </td>
  </tr>
  <tr>
    <td> 1 </td>
    <td> 262 </td>
    <td> 3182 </td>
  </tr>
</table>

<table>
  <tr>
    <th>Class</th>
    <th>Precision</th>
    <th>Recall</th>
  </tr>
  <tr>
    <th>0</th>
    <td>0.78</td>
    <td>0.71</td>
  </tr>
  <tr>
    <th>1</th>
    <td>0.89</td>
    <td>0.92</td>
  </tr>
</table>
F1 score = 0.908

## Udder segmentation model
See udder_labels for more details
* Yolo V8 = pt model
* settings
### Training, validation and test sets (sane as keypoints)
#### Number of cows in each set
<table>
  <tr>
    <th>Set</th>
    <th>A</th>
    <th>B</th>
    <th>C</th>
    <th>D</th>
    <th>Grand Total</th>
  </tr>
  <tr>
    <td>Train</td>
    <td>17</td>
    <td>20</td>
    <td>15</td>
    <td>15</td>
    <td>67</td>
  </tr>
  <tr>
    <td>Val</td>
    <td>6</td>
    <td>7</td>
    <td>5</td>
    <td>5</td>
    <td>23</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>6</td>
    <td>7</td>
    <td>5</td>
    <td>5</td>
    <td>23</td>
  </tr>
  <tr>
    <td>Total</td>
    <td>29</td>
    <td>34</td>
    <td>25</td>
    <td>25</td>
    <td>113</td>
  </tr>
</table>

#### Number of images in each set
<table>
  <tr>
    <th>Set</th>
    <th>A</th>
    <th>B</th>
    <th>C</th>
    <th>D</th>
    <th>Grand Total</th>
  </tr>
  <tr>
    <td>Train</td>
    <td>579</td>
    <td>700</td>
    <td>449</td>
    <td>450</td>
    <td>2178</td>
  </tr>
  <tr>
    <td>Val</td>
    <td>210</td>
    <td>245</td>
    <td>150</td>
    <td>150</td>
    <td>755</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>180</td>
    <td>245</td>
    <td>150</td>
    <td>150</td>
    <td>725</td>
  </tr>
  <tr>
    <td>Total</td>
    <td>969</td>
    <td>1190</td>
    <td>749</td>
    <td>750</td>
    <td>3658</td>
  </tr>
</table>

### Results
<table>
  <tr>
    <th>Metric</th>
    <th>Intersection over union</th>
  </tr>
  <tr>
    <td>Mean</td>
    <td>0.929</td>
  </tr>
  <tr>
    <td>Min</td>
    <td>0.735</td>
  </tr>
  <tr>
    <td>Max</td>
    <td>0.975</td>
  </tr>
  <tr>
    <td>Median</td>
    <td>0.938</td>
  </tr>
  <tr>
    <td>Std dev</td>
    <td>0.035</td>
  </tr>
</table>

<img src = "plots\pred_segments.png" width = 400>

## Teat keypoint detection model
See udder_labels for more details
* Yolo V8 = pt model
* settings
### Results
<table>
  <tr>
    <th>Metric</th>
    <th>Left front</th>
    <th>Right front</th>
    <th>Left back</th>
    <th>Right back</th>
  </tr>
  <tr>
    <td>Mean</td>
    <td>0.070</td>
    <td>0.079</td>
    <td>0.062</td>
    <td>0.077</td>
  </tr>
  <tr>
    <td>Min</td>
    <td>0.0018</td>
    <td>0.0018</td>
    <td>0.0015</td>
    <td>0.0025</td>
  </tr>
  <tr>
    <td>Max</td>
    <td>0.570</td>
    <td>0.542</td>
    <td>0.528</td>
    <td>0.620</td>
  </tr>
  <tr>
    <td>Median</td>
    <td>0.045</td>
    <td>0.058</td>
    <td>0.049</td>
    <td>0.060</td>
  </tr>
  <tr>
    <td>Std dev</td>
    <td>0.072</td>
    <td>0.069</td>
    <td>0.055</td>
    <td>0.065</td>
  </tr>
</table>

Distance is normalized by the diagonal of the udder box

<img src = "plots\pred_kp.png" width = 400>


