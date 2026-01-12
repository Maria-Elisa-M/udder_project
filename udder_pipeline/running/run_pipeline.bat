
call activate env-udder-00
python scripts\01_getdeptharrays_final.py udder_config.json
call conda deactivate

call activate udder
python scripts\02_predict_labels_final.py udder_config.json
call conda deactivate

call activate env-udder-03
python scripts\03_watershed_segment_final.py udder_config.json
call conda deactivate

call activate env-udder-04
python scripts\04_predict_class_ws_final.py udder_config.json
python scripts\05_write_good_frames_final.py udder_config.json
call conda deactivate

call activate env-udder-03
python scripts\06_features_shape_final.py udder_config.json
python scripts\07_lad_pointclouds_final.py udder_config.json
python scripts\08_features_volume_final.py udder_config.json
python scripts\09_fetaures_teat_final.py udder_config.json
python scripts\10_fetaure_table_final.py udder_config.json
call conda deactivate