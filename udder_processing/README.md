# Udder processing
Directory:
* pred_labels/
  * keypoints
  * bbox
  * segments
* validate_watershed/
  * pred_labels/
    * keypoints
    * bbox
    * segments
* watershed_segments_labeled
* watershed_segments_predicted

## Predict new cows
1. Classify frame
2. If good:
3. If segment exists
4. If keypoints exist

<img src = "diagram_udder.png" width = 300> 

Note:
* Frame index in labeled cows starts at 1.
* Frame index in predicted cows starts at 0. (My mistake, but it was too late to fix it)

## Watershed
* missing values in the depth map were filled in using the impaitning method from sk learn
* watershed had better results when the missing values were filled in compared to when missing values were kept

<table>
  <tr>
    <th>With NO missing values</th>
    <th>With NO missing values</th>
  </tr>
  <tr>
    <td><img src = "readme_examples/785_229.jpg" width = 380></td>
    <td><img src = "readme_examples/785_229_mv.jpg" width = 380></td>
  </tr>
  <tr>
    <td><img src = "readme_examples/785_782.jpg" width = 380></td>
    <td><img src = "readme_examples/785_782_mv.jpg" width = 380></td>
  </tr>
  <tr>
    <td><img src = "readme_examples/1281_757.jpg" width = 380></td>
    <td><img src = "readme_examples/1281_757_mv.jpg" width = 380></td>
  </tr>
</table>



# final scripts


# working scripts
