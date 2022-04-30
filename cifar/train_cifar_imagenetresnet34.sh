#!/bin/sh
### resnet
SCRIPT_FILE="train_cifar_all_methods.py"
echo ${SCRIPT_FILE}
python ./scripts/${SCRIPT_FILE} \
  --net_type imagenetresnet34 \
  --depth 56 \
  --width 4 \
  --dataset cifar100 \
  --batch_size 128 \
  --lr 0.1 \
  --method ols \
  --expname imagenetresnet34_ols \
  --wd 5e-4 \
  --epochs 300 \
  --workers 8 \
  --olsalpha 0.5 \

