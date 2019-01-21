from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils
from chainer import cuda


# Load Network

import dill

with open("_models/train_resnet_cifar10_gpu_(network).bn", "r") as f:
    branchyNet = dill.load(f)

# branchyNet.print_models()  # ren +


# Import Data

from datasets import pcifar10

print '1. pcifar10.get_data()'

_, _, x_test, y_test = pcifar10.get_data()


# Settings

TEST_BATCHSIZE = 64  # 1  #  ren -


print '2. set network to inference mode'

branchyNet.to_gpu()
branchyNet.testing()
branchyNet.verbose = False
branchyNet.gpu = True  # ren +


branchyNet.to_gpu()

g_baseacc, g_basediff, g_num_exits, g_accbreakdowns = utils.test(branchyNet, x_test, y_test, main=True,
                                                                 batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(y_test))) * 1000.

branchyNet.to_cpu()
with open("_models/test_resnet_cifar10_gpu_(g_baseacc).pkl", "w") as f:
    dill.dump({'g_baseacc': g_baseacc}, f)
with open("_models/test_resnet_cifar10_gpu_(g_basediff).pkl", "w") as f:
    dill.dump({'g_basediff': g_basediff}, f)
with open("_models/test_resnet_cifar10_gpu_(g_num_exits).pkl", "w") as f:
    dill.dump({'g_num_exits': g_num_exits}, f)
with open("_models/test_resnet_cifar10_gpu_(g_accbreakdowns).pkl", "w") as f:
    dill.dump({'g_accbreakdowns': g_accbreakdowns}, f)


# Specify thresholds


thresholds = [0.001, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 0.75]

print 'Thresholds : ', thresholds


print '3. utils.screen_branchy()'

branchyNet.to_gpu()

branchyNet.gpu = True  # ren +
branchyNet.verbose = True  # ren +

g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
                                                      batchsize=TEST_BATCHSIZE, verbose=True)

# g_ts, g_accs, g_diffs, g_exits = utils.screen_leaky(leakyNet, x_test, y_test, thresholds, inc_amt=-0.1,
#                                                     batchsize=TEST_BATCHSIZE, verbose=True)

#convert to ms
g_diffs *= 1000.

branchyNet.to_cpu()
with open("_models/test_resnet_cifar10_gpu_(g_ts).pkl", "w") as f:
    dill.dump({'g_ts': g_ts}, f)
with open("_models/test_resnet_cifar10_gpu_(g_accs).pkl", "w") as f:
    dill.dump({'g_accs': g_accs}, f)
with open("_models/test_resnet_cifar10_gpu_(g_diffs).pkl", "w") as f:
    dill.dump({'g_diffs': g_diffs}, f)
with open("_models/test_resnet_cifar10_gpu_(g_exits).pkl", "w") as f:
    dill.dump({'g_exits': g_exits}, f)
