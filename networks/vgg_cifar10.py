from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.links as L
import chainer.functions as F

def get_network(percentTrainKeeps=1):
    network = [

    self.conv1_1 = L.Convolution2D(None, 64, 3, pad=1)
    self.bn1_1 = L.BatchNormalization(64)
    self.conv1_2 = L.Convolution2D(64, 64, 3, pad=1)
    self.bn1_2 = L.BatchNormalization(64)

    self.conv2_1 = L.Convolution2D(64, 128, 3, pad=1)
    self.bn2_1 = L.BatchNormalization(128)
    self.conv2_2 = L.Convolution2D(128, 128, 3, pad=1)
    self.bn2_2 = L.BatchNormalization(128)

    self.conv3_1 = L.Convolution2D(128, 256, 3, pad=1)
    self.bn3_1 = L.BatchNormalization(256)
    self.conv3_2 = L.Convolution2D(256, 256, 3, pad=1)
    self.bn3_2 = L.BatchNormalization(256)
    self.conv3_3 = L.Convolution2D(256, 256, 3, pad=1)
    self.bn3_3 = L.BatchNormalization(256)
    self.conv3_4 = L.Convolution2D(256, 256, 3, pad=1)
    self.bn3_4 = L.BatchNormalization(256)

    self.fc4 = L.Linear(None, 1024)
    self.fc5 = L.Linear(1024, 1024)
    self.fc6 = L.Linear(1024, n_class)



        L.Convolution2D(1, 5, 5, stride=1, pad=3),
        Branch([L.Linear(None, 10)]),  # 1
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        Branch([L.Linear(None, 10)]),  # 2
        L.Convolution2D(5, 10, 5, stride=1, pad=3),
        Branch([L.Linear(None, 10)]),  # 3
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        Branch([L.Linear(None, 10)]),  # 4
        L.Convolution2D(10, 20, 5, stride=1, pad=3),
        Branch([L.Linear(None, 10)]),  # 5
        FL(F.max_pooling_2d, 2, 2),
        FL(F.ReLU()),
        Branch([L.Linear(None, 10)]),  # 6
        L.Linear(None, 84),  # 720, 84
        Branch([L.Linear(None, 10)])  # 84, 10
    ]
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
