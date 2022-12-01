dataset=aihub
# dataset=seaships_all

python WaSR/predict.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--weights /home/leeyoonji/workspace/git/WaSR/output/logs/wasr_resnet101_pretrained_mastr1478/20221110164130/checkpoints/epoch=61-step=18290.ckpt \
--output_dir WaSR/output/predictions/pretrained_mastr1478/20221110164130_best_$dataset \
--mode eval
# --mode pred