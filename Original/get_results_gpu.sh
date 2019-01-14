#!/bin/sh
../download_data.sh

# mkdir _models
# mkdir _figs

# python -u experiment_lenet_mnist_gpu.py
# python -u experiment_alex_cifar10_gpu.py
python -u experiment_resnet_cifar10_gpu.py