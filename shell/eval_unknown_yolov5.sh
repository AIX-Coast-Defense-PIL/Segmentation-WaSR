timestamp=`date +%Y%m%d%H%M%S`
dataset=testdata
version=exp2
output_dir=WaSR/output/predictions/unknown/$version\_$dataset\_$timestamp
mkdir -p $output_dir

python WaSR/eval_unknown_v5.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--wasr_model wasr_resnet101 \
--wasr_weights WaSR/output/logs/wasr_resnet101_pretrained_mastr1478/20221110164130/checkpoints/epoch=61-step=18290.ckpt \
--yolo_weights yolov5/runs/train/$version/weights/best.pt \
--output_dir $output_dir \
--batch_size 1