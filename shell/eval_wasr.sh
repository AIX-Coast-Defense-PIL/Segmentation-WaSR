dataset=wasr_all

python WaSR/eval.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--weights WaSR/output/logs/wasr_resnet101_pretrained_wasr_all_crop/20221010225445/weights.pt \
--output_dir WaSR/output/predictions/20221010225445_$dataset