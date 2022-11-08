python ../PyTorch_YOLOv4/detect.py \
--weights /home/leeyoonji/workspace/PyTorch_YOLOv4/nexreal/best.pt \
--source /home/leeyoonji/workspace/git/datasets/미상물체/log_on_the_sea-Google \
--cfg /home/leeyoonji/workspace/PyTorch_YOLOv4/nexreal/nipa13.cfg \
--names /home/leeyoonji/workspace/PyTorch_YOLOv4/nexreal/nipa_cls13.names \
--output /home/leeyoonji/workspace/PyTorch_YOLOv4/inference/output_google \
--save-txt
# --conf-thres 0.05