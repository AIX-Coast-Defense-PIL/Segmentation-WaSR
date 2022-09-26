python WaSR/train.py \
--train_config WaSR/configs/wasr_all_train.yaml \
--val_config WaSR/configs/wasr_all_val.yaml \
--model_name my_wasr_no_imu \
--model wasr_resnet101 \
--pretrained False \
--validation \
--batch_size 4 \
--epochs 30 \
--output_dir WaSR/output