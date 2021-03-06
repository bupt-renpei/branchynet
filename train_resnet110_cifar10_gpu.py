from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils
from chainer import cuda


# Define Network

from networks import resnet_cifar
branchyNet = resnet_cifar.get_network(n_class=10, n_layer=18)

# branchyNet.print_models()  # ren +
print '0. Define Network'

branchyNet.to_gpu()
branchyNet.training()


# Import Data

# from datasets import pcifar10  # original
# x_train, y_train, _, _ = pcifar10.get_data()  # original

from _tool import chainerDataset
x_train, y_train, _, _ = chainerDataset.get_chainer_cifar10()

print '1. Import Data'


# Settings

TRAIN_BATCHSIZE = 16  # 64
# TEST_BATCHSIZE = 64  # 1  #  ren -
TRAIN_NUM_EPOCHS = 100

branchyNet.verbose = False  # ren +
branchyNet.gpu = True  # ren +


# Train Main Network

print '2. Train Main Network'

main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)

# print 'main_loss : ', main_loss, ' | main_acc : ', main_acc, ' | main_time : ', main_time


# Train BranchyNet

print '3. Train BranchyNet'

TRAIN_NUM_EPOCHS = 100

branch_loss, branch_acc, branch_time = utils.train(branchyNet, x_train, y_train, batchsize=TRAIN_BATCHSIZE,
                                                   num_epoch=TRAIN_NUM_EPOCHS)

# print 'branch_loss : ', branch_loss, ' | branch_acc : ', branch_acc, ' | branch_time : ', branch_time


print '4. Save model/data'

import dill
branchyNet.to_cpu()
with open("_models/train_resnet110_cifar10_gpu_(network).bn", "w") as f:
    dill.dump(branchyNet, f)
with open("_models/train_resnet110_cifar10_gpu_(main_loss).pkl", "w") as f:
    dill.dump({'main_loss': main_loss}, f)
with open("_models/train_resnet110_cifar10_gpu_(main_acc).pkl", "w") as f:
    dill.dump({'main_acc': main_acc}, f)
with open("_models/train_resnet110_cifar10_gpu_(main_time).pkl", "w") as f:
    dill.dump({'main_time': main_time}, f)
with open("_models/train_resnet110_cifar10_gpu_(branch_loss).pkl", "w") as f:
    dill.dump({'branch_loss': branch_loss}, f)
with open("_models/train_resnet110_cifar10_gpu_(branch_acc).pkl", "w") as f:
    dill.dump({'branch_acc': branch_acc}, f)
with open("_models/train_resnet110_cifar10_gpu_(branch_time).pkl", "w") as f:
    dill.dump({'branch_time': branch_time}, f)
