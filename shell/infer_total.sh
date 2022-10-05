python WaSR/predict_obs.py \
--dataset_config WaSR/configs/seaships_test.yaml \
--model wasr_resnet101 \
--wasr_weights WaSR/output/logs/my_wasr_no_imu/version_13/weights.pth \
--yolo_weights yolov5/runs/train/exp2/weights/best.pt \
--output_dir WaSR/output/predictions/test