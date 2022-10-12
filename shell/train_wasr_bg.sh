timestamp=`date +%Y%m%d%H%M%S`
dataset=mastr1325
model=wasr_resnet101

model_name=$model\_pretrained_$dataset
log_dir=WaSR/output/logs/$model_name/$timestamp
mkdir -p $log_dir

python WaSR/train.py \
--train_config WaSR/configs/$dataset\_train.yaml \
--val_config WaSR/configs/$dataset\_val.yaml \
--model $model \
--model_name $model_name \
--pretrained True \
--validation \
--batch_size 4 \
--epochs 50 \
--output_dir WaSR/output \
--datetime $timestamp &>> $log_dir/$model_name\_$timestamp.log