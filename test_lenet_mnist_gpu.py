from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils
from chainer import cuda


# Define Network

from networks import lenet_mnist

branchyNet = lenet_mnist.get_network()

import dill

with open("_models/train_lenet_mnist_gpu_(network).bn", "r") as f:
    dill.load(branchyNet, f)

branchyNet.print_models()  # ren +

