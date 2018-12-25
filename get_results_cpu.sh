#!/bin/sh
./download_data.sh

#mkdir _models
#mkdir _figs

python experiment_lenet_mnist_cpu.py
python experiment_alex_cifar10_cpu.py
python experiment_resnet_cifar10_cpu.py
