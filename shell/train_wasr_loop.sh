dataset=mastr1478
model=wasr_resnet101
model_name=$model\_pretrained_$dataset

# separation_loss=("wsl" "cwsl" "cosl" "cwssl" "cwosl" "hwssl" "hwosl" "cwskyl" "coskyl" "cskyol" "cskywl")
separation_loss=("cskyol" "cskywl")
len_separation_loss=${#separation_loss[@]}

for ((loss=0;loss<$len_separation_loss;loss++)) do
    timestamp=`date +%Y%m%d%H%M%S`
    log_dir=WaSR/output/logs/$model_name/$timestamp
    mkdir -p $log_dir
    
    CUDA_VISIBLE_DEVICES=1 python WaSR/train.py \
    --train_config WaSR/configs/$dataset\_train.yaml \
    --val_config WaSR/configs/$dataset\_val.yaml \
    --model $model \
    --model_name $model_name \
    --workers 2 \
    --validation \
    --batch_size 4 \
    --epochs 100 \
    --separation_loss ${separation_loss[loss]} \
    --output_dir WaSR/output \
    --datetime $timestamp &>> $log_dir/$model_name\_$timestamp.log
done