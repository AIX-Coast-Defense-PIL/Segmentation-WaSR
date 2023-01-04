dataset=aihub
# dataset=seaships_all

python WaSR/predict.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--weights /home/leeyoonji/workspace/git/WaSR/nex_logs/wasr_resnet101_pretrained_mastr1478/20221204232114/checkpoints/cwosl_epoch=30-step=2262.ckpt \
--output_dir WaSR/output/predictions/pretrained_mastr1478/20221204232114_nex_cwosl_$dataset \
--mode eval
# --mode pred