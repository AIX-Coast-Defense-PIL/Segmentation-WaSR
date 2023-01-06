timestamp=`date +%Y%m%d%H%M%S`
dataset=testdata
version=loss_cowl
output_dir=WaSR/output/predictions_unknown/$version\_$dataset\_$timestamp
mkdir -p $output_dir

python WaSR/eval_unknown_yolov4.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--wasr_model wasr_resnet101 \
--wasr_weights /home/leeyoonji/workspace/git/WaSR/output_loss/logs/wasr_resnet101_loss_wasr_all/20230106063110_cosl/checkpoints/epoch=26-step=3698.ckpt \
--yolo_weights yolov4/weights/2022-11-17/best.pt \
--yolo_cfg yolov4/weights/2022-11-17/nipa_mish10.cfg \
--output_dir $output_dir \
--batch_size 1 &>> $output_dir/$version\_$timestamp.log