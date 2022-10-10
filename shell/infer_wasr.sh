dataset=seaships

python WaSR/predict.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--weights WaSR/old_output/logs/my_wasr_no_imu/version_10/weights.pth \
--output_dir WaSR/output/predictions/my_wasr_no_imu_10_$dataset
