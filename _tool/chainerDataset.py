import numpy
from chainer import datasets
import random

# Import Data

# BranchyNet (original)


# from datasets import pcifar10  # original  # ren -
# x_train,y_train,x_test,y_test = pcifar10.get_data()  # original # ren -

# print x_train.shape  # (50000, 3, 32, 32)
# print y_train.shape  # (50000,)
# print type(x_test)   # (10000, 3, 32, 32)
# print type(y_test)   # (10000,)

def get_lenet_mnist():
    raw_train, raw_test = datasets.get_mnist(withlabel=True, ndim=1, scale=1.)
    return process_data(raw_train, raw_test)


def get_chainer_cifar10():
    raw_train, raw_test = datasets.get_cifar10(withlabel=True, ndim=3, scale=1.)
    return process_data(raw_train, raw_test)


def get_chainer_cifar100():
    raw_train, raw_test = datasets.get_cifar100(withlabel=True, ndim=3, scale=1.)
    return process_data(raw_train, raw_test)

# raw_train,raw_test = datasets.get_cifar100(withlabel=True, ndim=3, scale=1.)

# print len(raw_train) # 50000
# print len(raw_test) # 10000
# print len(raw_train[0]) # 2
# print len(raw_train[1]) # 2
# print raw_train[0][0].shape
# print raw_train[0][1]


def process_data(raw_train, raw_test):
    list_raw_train_x = []
    list_raw_train_y = []
    list_raw_test_x = []
    list_raw_test_y = []

    # for item_train in raw_train:
    #     list_raw_train_x.append(item_train[0])
    #     list_raw_train_y.append(item_train[1])
    #
    # for item_test in raw_test:
    #     list_raw_test_x.append(item_test[0])
    #     list_raw_test_y.append(item_test[1])

    # Train-sample
    for i in random.sample(range(1, 99), 10):
        list_raw_train_x.append(raw_train[i][0])
        list_raw_train_y.append(raw_train[i][1])

    # Test-sample
    for i in random.sample(range(1, 99), 10):
        list_raw_test_x.append(raw_test[i][0])
        list_raw_test_y.append(raw_test[i][1])

    x_train = numpy.array(list_raw_train_x)
    y_train = numpy.array(list_raw_train_y)
    x_test = numpy.array(list_raw_test_x)
    y_test = numpy.array(list_raw_test_y)

    print x_train.shape
    print y_train.shape
    print x_test.shape
    print y_test.shape

    return x_train, y_train, x_test, y_test


