#!/bin/bash
python -u train_lenet_mnist_gpu.py

python -u train_alex_cifar100_gpu.py
python -u train_alex_cifar10_gpu.py

python -u train_vgg16_cifar100_gpu.py
python -u train_vgg16_cifar10_gpu.py

python -u train_resnet32_cifar100_gpu.py
python -u train_resnet32_cifar10_gpu.py

python -u train_resnet56_cifar100_gpu.py
python -u train_resnet56_cifar10_gpu.py

# python -u train_resnet110_cifar100_gpu.py
# python -u train_resnet110_cifar10_gpu.py