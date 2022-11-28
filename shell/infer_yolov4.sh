python yolov4/detect.py \
--weights yolov4/weights/2022-11-17/best.pt \
--source datasets/SeaShips/images/test \
--cfg yolov4/weights/2022-11-17/nipa_mish10.cfg \
--names yolov4/weights/2022-11-17/nipa_cls10.names \
--output yolov4/inference/output_seaships_1117 \
--save-txt
# --conf-thres 0.05