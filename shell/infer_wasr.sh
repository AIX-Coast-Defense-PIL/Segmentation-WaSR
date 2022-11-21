dataset=aihub
# dataset=seaships_all

python WaSR/predict.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--weights WaSR/output/logs/wasr_resnet101_pretrained_mastr1478/20221110164130/weights.pt \
--output_dir WaSR/output/predictions/test2 \
--mode eval
# --mode pred
