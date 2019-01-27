#!/bin/bash
echo "lenet"
python -u test_lenet_mnist_gpu.py

echo "alex cifar100"
python -u test_alex_cifar100_gpu.py
echo "alex cifar10"
python -u test_alex_cifar10_gpu.py

echo "vgg16 cifar100"
python -u test_vgg16_cifar100_gpu.py
echo "vgg16 cifar10"
python -u test_vgg16_cifar10_gpu.py

echo "resnet32 cifar100"
python -u test_resnet32_cifar100_gpu.py
echo "resnet32 cifar10"
python -u test_resnet32_cifar10_gpu.py

echo "resnet56 cifar100"
python -u test_resnet56_cifar100_gpu.py
echo "resnet56 cifar10"
python -u test_resnet56_cifar10_gpu.py

echo "resnet110 cifar100"
python -u train_resnet110_cifar100_gpu.py
# python -u train_resnet110_cifar10_gpu.py