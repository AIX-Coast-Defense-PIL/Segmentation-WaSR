timestamp=`date +%Y%m%d%H%M%S`
dataset=testdata
version=exp2
output_dir=WaSR/output/predictions/unknown/$version\_$dataset\_$timestamp
mkdir -p $output_dir

python WaSR/eval_unknown.py \
--wasr_dataset WaSR/configs/$dataset\_test.yaml \
--wasr_model wasr_resnet101 \
--wasr_weights WaSR/old_output/logs/my_wasr_no_imu/version_13/weights.pth \
--output_dir $output_dir \
--batch_size 1 \
--single-cls \
--yolo_dataset yolov5/data/$dataset.yaml \
--yolo_weights yolov5/runs/train/$version/weights/best.pt &>> $output_dir/$version\_$dataset\_$timestamp.log
