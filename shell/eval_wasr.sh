dataset=seaships

python WaSR/eval.py \
--dataset_config WaSR/configs/$dataset\_test.yaml \
--model wasr_resnet101 \
--weights WaSR/output/logs/wasr_resnet101_pretrained_wasrALL/20221009171743/weights.pt \
--output_dir WaSR/output/predictions/20221009171743_$dataset