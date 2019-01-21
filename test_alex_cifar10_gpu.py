from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils
from chainer import cuda


# Load Network

import dill
with open("_models/train_alexnet_cifar10_gpu_(network).bn", "r") as f:
    branchyNet = dill.load(f)

# branchyNet.print_models()  # ren +
print '0. Load Network'


# Import Data

# from datasets import pcifar10  # original
# _, _, x_test, y_test = pcifar10.get_data()  # original

from _tool import chainerDataset
_, _, x_test, y_test = chainerDataset.get_chainer_cifar10()

print '1. Import Data'


# Settings

TEST_BATCHSIZE = 128


print '2. set network to inference mode'

branchyNet.to_gpu()
branchyNet.testing()
branchyNet.verbose = False
branchyNet.gpu = True  # ren +


branchyNet.to_gpu()

g_baseacc, g_basediff, g_num_exits, g_accbreakdowns = utils.test(branchyNet, x_test, y_test, main=True,
                                                                 batchsize=TEST_BATCHSIZE)

print 'main accuracy : ', g_baseacc

g_basediff = (g_basediff / float(len(y_test))) * 1000.

branchyNet.to_cpu()
with open("_models/test_alexnet_cifar10_gpu_(g_baseacc).pkl", "w") as f:
    dill.dump({'g_baseacc': g_baseacc}, f)
with open("_models/test_alexnet_cifar10_gpu_(g_basediff).pkl", "w") as f:
    dill.dump({'g_basediff': g_basediff}, f)
with open("_models/test_alexnet_cifar10_gpu_(g_num_exits).pkl", "w") as f:
    dill.dump({'g_num_exits': g_num_exits}, f)
with open("_models/test_alexnet_cifar10_gpu_(g_accbreakdowns).pkl", "w") as f:
    dill.dump({'g_accbreakdowns': g_accbreakdowns}, f)


b_baseacc, b_basediff, b_num_exits, b_accbreakdowns = utils.test(branchyNet, x_test, y_test,
                                                                 batchsize=TEST_BATCHSIZE)

print b_basediff, b_num_exits