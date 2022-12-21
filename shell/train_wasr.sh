python WaSR/train.py \
--train_config WaSR/configs/mastr1325_train.yaml \
--val_config WaSR/configs/mastr1325_val.yaml \
--model wasr_resnet101 \
--model_name wasr_resnet101_mastr1325 \
--workers 2 \
--validation \
--batch_size 4 \
--epochs 100 \
--output_dir WaSR/output