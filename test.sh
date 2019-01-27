#!/bin/bash
python -u test_lenet_mnist_gpu.py

python -u test_alex_cifar100_gpu.py
python -u test_alex_cifar10_gpu.py

python -u test_vgg16_cifar100_gpu.py
python -u test_vgg16_cifar10_gpu.py

python -u test_resnet32_cifar100_gpu.py
python -u test_resnet32_cifar10_gpu.py

python -u test_resnet56_cifar100_gpu.py
python -u test_resnet56_cifar10_gpu.py

# python -u train_resnet110_cifar100_gpu.py
# python -u train_resnet110_cifar10_gpu.py