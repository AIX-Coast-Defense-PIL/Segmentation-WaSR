timestamp=`date +%Y%m%d%H%M%S`
dataset=aihub

python WaSR/predict_unknown.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--wasr_weights WaSR/output/logs/wasr_resnet101_pretrained_wasr_all_v2/20221116184846/weights.pt \
--yolo_weights yolov5/runs/train/exp2/weights/best.pt \
--output_dir WaSR/output/predictions/unknown/pred_$dataset\_$timestamp \
--batch_size 1 \
--imgsz 384 512