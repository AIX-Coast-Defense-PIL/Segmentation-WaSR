dataset=seaships

python WaSR/predict.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--weights WaSR/output/logs/wasr_resnet101_scratch_mastr1478/20221111141545/weights.pt \
--output_dir WaSR/output/predictions/scratch_mastr1478/20221111141545_$dataset
