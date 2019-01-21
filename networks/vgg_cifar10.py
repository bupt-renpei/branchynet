from __future__ import absolute_import

from branchynet.links.links import *
from branchynet.net import BranchyNet

import chainer.links as L
import chainer.functions as F

# VGG-16

def get_network(percentTrainKeeps=1):
    network = [
        L.Convolution2D(None, 64, 3, stride=1, pad=1),
        L.Convolution2D(64, 64, 3, stride=1, pad=1),
        FL(F.max_pooling_2d, 2, 2),

        L.Convolution2D(64, 128, 3, stride=1, pad=1),
        L.Convolution2D(128, 128, 3, stride=1, pad=1),
        Branch([L.Linear(None, 10)]),  # 1
        FL(F.max_pooling_2d, 2, 2),

        L.Convolution2D(128, 256, 3, stride=1, pad=1),
        L.Convolution2D(256, 256, 3, stride=1, pad=1),
        L.Convolution2D(256, 256, 3, stride=1, pad=1),
        FL(F.max_pooling_2d, 2, 2),

        L.Convolution2D(256, 512, 3, stride=1, pad=1),
        L.Convolution2D(512, 512, 3, stride=1, pad=1),
        L.Convolution2D(512, 512, 3, stride=1, pad=1),
        FL(F.max_pooling_2d, 2, 2),

        L.Convolution2D(512, 512, 3, stride=1, pad=1),
        L.Convolution2D(512, 512, 3, stride=1, pad=1),
        L.Convolution2D(512, 512, 3, stride=1, pad=1),
        FL(F.max_pooling_2d, 2, 2),

        L.Linear(None, 4096),
        FL(F.ReLU()),
        SL(FL(F.dropout, 0.5, train=True)),

        L.Linear(4096, 4096),
        FL(F.ReLU()),
        SL(FL(F.dropout, 0.5, train=True)),

        #L.Linear(4096, 1000),
        Branch([L.Linear(None, 10)])


    ]
    net = BranchyNet(network, percentTrainKeeps=percentTrainKeeps)
    return net
