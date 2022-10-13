timestamp=`date +%Y%m%d%H%M%S`
dataset=testdata
version=exp2

python yolov5/val.py \
--weights yolov5/runs/train/$version/weights/best.pt \
--data $dataset.yaml \
--save-txt \
--save-hybrid \
--save-conf \
--task test &>> yolov5/runs/val/logs/$version\_$dataset\_$timestamp.log
