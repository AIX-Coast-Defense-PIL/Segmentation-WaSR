timestamp=`date +%Y%m%d%H%M%S`
dataset=testdata
version=brightness_loss
output_dir=WaSR/output/inference_unknown/$version\_$dataset\_$timestamp
mkdir -p $output_dir

python WaSR/predict_unknown.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--wasr_weights /home/leeyoonji/workspace/git/WaSR/output_brightness/logs/wasr_resnet101_brightness_preprocess/brightness_loss/checkpoints/epoch=53-step=7397.ckpt \
--yolo_weights yolov4/weights/2022-11-17/best.pt \
--yolo_cfg yolov4/weights/2022-11-17/nipa_mish10.cfg \
--output_dir $output_dir \
--imgsz 640 \
--batch_size 1 &>> $output_dir/$version\_$timestamp.log