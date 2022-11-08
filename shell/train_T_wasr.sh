python WaSR-T/train.py \
--train-config WaSR-T/configs/mastr1325_train.yaml \
--val-config WaSR-T/configs/mastr1325_val.yaml \
--model-name my_wasr_no_imu \
--model wasr_resnet101 \
--pretrained False \
--validation \
--batch-size 4 \
--epochs 30 \
--output-dir WaSR-T/output