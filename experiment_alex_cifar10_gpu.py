from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils
from chainer import cuda


# Define Network

from networks import alex_cifar10

branchyNet = alex_cifar10.get_network()
branchyNet.to_gpu()
branchyNet.training()


# Import Data

from datasets import pcifar10

print '1. pcifar10.get_data()'

x_train,y_train,x_test,y_test = pcifar10.get_data()


# Settings

TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 1
TRAIN_NUM_EPOCHS = 50


# Train Main Network

print '2. Train Main Network'

main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)


# Train BranchyNet

print '3. Train BranchyNet'

TRAIN_NUM_EPOCHS = 100

branch_loss, branch_acc, branch_time = utils.train(branchyNet, x_train, y_train, batchsize=TRAIN_BATCHSIZE,
                                                   num_epoch=TRAIN_NUM_EPOCHS)


# set network to inference mode

# branchyNet.testing()


# Visualizing Network Training

# (Visualizing) visualize.plot_layers(main_loss, xlabel='Epochs', ylabel='Training Loss')
# (Visualizing) visualize.plot_layers(main_acc, xlabel='Epochs', ylabel='Training Accuracy')

# (Visualizing) visualize.plot_layers(zip(*branch_loss), xlabel='Epochs', ylabel='Training Loss')
# (Visualizing) visualize.plot_layers(zip(*branch_acc), xlabel='Epochs', ylabel='Training Accuracy')


# Run test suite and visualize

print '4. set network to inference mode'

# set network to inference mode

branchyNet.testing()
branchyNet.verbose = False

branchyNet.to_gpu()
g_baseacc, g_basediff, _, _ = utils.test(branchyNet, x_test, y_test, main=True, batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(y_test))) * 1000.


# (CPU) branchyNet.to_cpu()
# (CPU) c_baseacc, c_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
# (CPU) c_basediff = (c_basediff / float(len(y_test))) * 1000.


# Specify thresholds

# thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.75, 1., 5., 10.]
thresholds = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5]



# GPU

print '5. utils.screen_branchy()'

branchyNet.to_gpu()
g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
                                                      batchsize=TEST_BATCHSIZE, verbose=True)
# g_ts, g_accs, g_diffs, g_exits = utils.screen_leaky(leakyNet, x_test, y_test, thresholds, inc_amt=-0.1,
#                                                     batchsize=TEST_BATCHSIZE, verbose=True)

# convert to ms
g_diffs *= 1000.


# (GPU-Visualizing) visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff, all_samples=False, inc_amt=-0.0001000,
# (GPU-Visualizing)                              our_label='BranchyAlexNet', orig_label='AlexNet', xlabel='Runtime (ms)',
# (GPU-Visualizing)                              title='AlexNet GPU', output_path='_figs/alexnet_gpu.pdf')


# CPU
# (CPU) branchyNet.to_cpu()
# (CPU) c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
# (CPU)                                                      batchsize=TEST_BATCHSIZE, verbose=True)
# c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, g_ts, inc_amt=0.01,
#                                                      batchsize=TEST_BATCHSIZE, prescreen=False, verbose=True)

# convert to ms
# (CPU) c_diffs *= 1000.


# (CPU-Visualizing) visualize.plot_line_tradeoff(c_accs, c_diffs, c_ts, c_exits, c_baseacc, c_basediff, all_samples=False, inc_amt=-0.0001000,
# (CPU-Visualizing)                              our_label='BranchyAlexNet', orig_label='AlexNet', xlabel='Runtime (ms)',
# (CPU-Visualizing)                              title='AlexNet CPU', output_path='_figs/alexnet_cpu.pdf')


# Compute table results
# (C-GPU) utils.branchy_table_results(c_baseacc, c_basediff, g_basediff, c_accs, c_diffs, g_accs, g_diffs, inc_amt=0.000,
# (C-GPU)                           network='AlexNet')


# Save model/data

print '6. Save model/data'

import dill
branchyNet.to_cpu()
with open("_models/alexnet_cifar10_GPU.bn", "w") as f:
    dill.dump(branchyNet, f)
with open("_models/alexnet_cifar10_results_GPU.pkl", "w") as f:
    dill.dump({'main_loss': main_loss, 'main_acc': main_acc, 'main_time': main_time, 'branch_loss': branch_loss,
               'branch_acc': branch_acc, 'branch_time': branch_time, 'g_baseacc': g_baseacc, 'g_basediff': g_basediff,
               'g_ts': g_ts, 'g_accs': g_accs, 'g_diffs': g_diffs, 'g_exits': g_exits}, f)
