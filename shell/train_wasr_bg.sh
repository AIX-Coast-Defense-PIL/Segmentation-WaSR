model_name="my_wasr_no_imu"
timestamp=`date +%Y%m%d%H%M%S`

python WaSR/train.py \
--train_config WaSR/configs/mastr1325_train.yaml \
--val_config WaSR/configs/mastr1325_val.yaml \
--model_name $model_name \
--model wasr_resnet101 \
--pretrained False \
--validation \
--batch_size 4 \
--epochs 1 \
--output_dir WaSR/output &>> WaSR/output/runs/$model_name\_"$timestamp".log