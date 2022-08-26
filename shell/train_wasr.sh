python WaSR/train.py \
--train_config WaSR/configs/mastr1325_train.yaml \
--val_config WaSR/configs/mastr1325_val.yaml \
--model_name my_wasr_no_imu \
--model wasr_resnet101 \
--pretrained False \
--validation \
--batch_size 4 \
--epochs 30 \
--output_dir WaSR/output