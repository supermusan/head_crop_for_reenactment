python crop_dperfs1.py --workers 15 --device_ids 0,1,2,3 --no-estimate-bbox --bbox_folder bbox-dperfs1 --format .png --neighbor_or_initial initial --out_folder /data_activate/wangweisen/reenactment/dataset/DeeperForensics-1.0/dperfs1-initial-png-0.25 --chunks_metadata dperfs1-initial-metadata-0.25.csv --increase 0.25
# python crop_timit.py --workers 15 --device_ids 0,1,2,3 --no-estimate-bbox --bbox_folder bbox-timit --format .png --neighbor_or_initial initial --out_folder timit-initial-png --chunks_metadata timit-initial-metadata.csv --min_size 200 --increase 0.25