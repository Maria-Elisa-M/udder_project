Collection date: 2023-11-17
Farm: Laufenberg
Milking parlour: VMS (two left: vms_1, vms_2, and two right:vms_3, vms_4)

# Folder structure
There is a folder for each computer used for video collcetion
-videos_1 -> Maria's laptop -> vms_3 and vms_4
-videos_2 -> Guilherme's laptop -> vms_1, vms_2, vms_3 and vms_4
-videos_3 -> Lab's pink laptop -> vms_1 and vms_2

# File namimg convention
<cowID>_<date>_<filenumber>.bag
example: 1223_20231117_153008.bag

Note: cow IDs were verified by time and robot with the delpro report
These file names were modified to have the correct cow ID
original cow ID -> corrected cow ID
-1332 -> 1432
-1179 -> 1279
-732 -> 723
-904 -> 1274

One video had no cow ID and was named 9999
After verification with delpro and the colored videos this cow was determined to be 1223
-9999 -> 1223

files were renamed as follows: (see script rename_files.ipynb)
videos_1/
9999_20231117_153008.bag -> 1223_20231117_153008.bag
1332_20231117_133858.bag -> 1432_20231117_133858.bag

videos_2/
732_20231117_175039.bag -> 723_20231117_175039.bag
1179_20231117_161747.bag -> 1279_20231117_161747.bag

videos_3/
904_1274_20231117_105454.bag -> 1274_20231117_105454.bag

# Ignored videos
The following videos produced an errow when trying to extact the frames

videos_1/
1184_20231117_172549.bag
20231117_153008.bag
738_20231117_104922.bag

videos_2/
855_20231117_170701.bag


# later for prediction this video was also ignored (produced an error)
videos_1/
1489_20231117_165935.bag