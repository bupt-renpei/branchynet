from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils
from chainer import cuda


import dill

with open("_models/train_alexnet_cifar10_gpu_(network).bn", "r") as f:
    branchyNet = dill.load(f)

branchyNet.print_models()  # ren +


# Import Data

from datasets import pcifar10

print '1. pcifar10.get_data()'

_, _, x_test, y_test = pcifar10.get_data()


TEST_BATCHSIZE = 64  # 1  #  ren -



print '5. set network to inference mode'

# set network to inference mode

branchyNet.to_gpu()
branchyNet.testing()
branchyNet.verbose = False
branchyNet.gpu = True  # ren +


branchyNet.to_gpu()

g_baseacc, g_basediff, g_num_exits, g_accbreakdowns = utils.test(branchyNet, x_test, y_test, main=True,
                                                                 batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(y_test))) * 1000.

branchyNet.to_cpu()
with open("_models/alexnet_cifar10_results_GPU_(Test).pkl", "w") as f:
    dill.dump({'g_baseacc': g_baseacc, 'g_basediff': g_basediff, 'g_num_exits': g_num_exits,
               'g_accbreakdowns': g_accbreakdowns}, f)