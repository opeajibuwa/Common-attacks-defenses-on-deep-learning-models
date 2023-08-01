#!/bin/bash

# Train models
python train.py --model lenet_mnist --epochs 50 --lr 0.3 --batchsz 64 >> train.log 2>&1
python train.py --model lenet_cifar10 --epochs 50 --lr 0.001 --optim adam >> train.log 2>&1
python train.py --model resnet_mnist --epochs 50 --lr 0.01 --optim adam --batchsz 128 >> train.log 2>&1
python train.py --model resnet_cifar10 --epochs 50 --lr 0.01 >> train.log 2>&1
python train.py --model resnet_dropout --epochs 50 --lr 0.01 >> train.log 2>&1
python train.py --model vgg_cifar10 --epochs 50 --lr 0.01 batchsz 128 >> train.log 2>&1
python train.py --model vgg_mnist --epochs 50 --lr 0.001 --batchsz 64 >> train.log 2>&1
python train.py --model resnet_cifar10_wd --epochs 50 --lr 0.1 --wd 0.00001 >> train.log 2>&1 

# Validate models
python valid.py --model lenet_cifar10 >> train.log 2>&1
python valid.py --model lenet_mnist >> train.log 2>&1
python valid.py --model resnet_cifar10 >> train.log 2>&1
python valid.py --model resnet_mnist >> train.log 2>&1
python valid.py --model vgg_mnist >> train.log 2>&1
python valid.py --model vgg_cifar10 >> train.log 2>&1
python valid.py --model resnet_wd >> train.log 2>&1
python valid.py --model lenet_mnist_adv >> train.log 2>&1
python valid.py --model resnet_cifar10_adv >> train.log 2>&1