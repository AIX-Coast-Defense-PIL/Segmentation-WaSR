python WaSR/predict.py \
--dataset_config WaSR/configs/google_test.yaml \
--model wasr_resnet101 \
--weights WaSR/output/logs/my_wasr_no_imu/version_10/weights.pth \
--output_dir WaSR/output/predictions/my_wasr_no_imu_google