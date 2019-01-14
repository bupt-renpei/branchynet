from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda


# Define Network

# print 'chainer.__version__ : ', chainer.__version__

from networks import lenet_mnist

print '1. lenet_mnist.get_network()'

branchyNet = lenet_mnist.get_network()
branchyNet.to_gpu()
branchyNet.training()


# Import Data

print '2. mnist.get_data()'

from datasets import mnist
x_train, y_train, x_test, y_test = mnist.get_data()


# Settings

TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 1
TRAIN_NUM_EPOCHS = 50


# Train Main Network

print '3. train Main Network (Start)'

main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)

print '3. train Main Network (Finish)'

# Train BranchyNet

print '4. train BranchyNet Network (Start)'

TRAIN_NUM_EPOCHS = 100
branch_loss, branch_acc, branch_time = utils.train(branchyNet, x_train, y_train, batchsize=TRAIN_BATCHSIZE,
                                                   num_epoch=TRAIN_NUM_EPOCHS)

print '4. train BranchyNet Network (Finish)'


# Save model/data

print '5. Save model (Start)'

import dill
branchyNet.to_cpu()
# (GPU) branchyNet.to_gpu()
with open("_models/lenet_mnist.bn", "w") as f:
    dill.dump(branchyNet, f)
with open("_models/lenet_mnist_gpu_results(train).pkl", "w") as f:
    dill.dump({'main_loss': main_loss, 'main_acc': main_acc, 'main_time': main_time, 'branch_loss': branch_loss,
               'branch_acc': branch_acc, 'branch_time': branch_time}, f)

print '5. Save mode (Finish)'


# set network to inference mode

# (redundancy) branchyNet.testing()


# Visualizing Network Training

# (visualize) visualize.plot_layers(main_loss, xlabel='Epochs', ylabel='Training Loss')
# (visualize) visualize.plot_layers(main_acc, xlabel='Epochs', ylabel='Training Accuracy')


# (visualize) visualize.plot_layers(zip(*branch_loss), xlabel='Epochs', ylabel='Training Loss')
# (visualize) visualize.plot_layers(zip(*branch_acc), xlabel='Epochs', ylabel='Training Accuracy')


# Run test suite and visualize

# set network to inference mode


# (INFERENCE) print '6. Set network to inference mode'

# (INFERENCE) branchyNet.testing()
# (INFERENCE) branchyNet.verbose = False

# (INFERENCE) branchyNet.to_gpu()
# (INFERENCE) g_baseacc, g_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
# (INFERENCE) g_basediff = (g_basediff / float(len(y_test))) * 1000.

# (INFERENCE) print '  g_baseacc : ', g_baseacc, 'g_basediff : ', g_basediff

# branchyNet.to_cpu()
# (CPU) c_baseacc, c_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
# (CPU) c_basediff = (c_basediff / float(len(y_test))) * 1000.



# Specify thresholds
# (INFERENCE) thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 3., 5., 10.]



#GPU
# (INFERENCE) branchyNet.to_cpu()
# (INFERENCE) g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds, batchsize=TEST_BATCHSIZE,
# (INFERENCE)                                                       verbose=True)

# (INFERENCE) print ' g_accs : ', g_accs

# g_ts, g_accs, g_diffs, g_exits = utils.screen_leaky(leakyNet, x_test, y_test, thresholds, inc_amt=-0.1,
#                                                     batchsize=TEST_BATCHSIZE, verbose=True)

#convert to ms
# (INFERENCE) g_diffs *= 1000.



# (visualize) visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff, all_samples=False, inc_amt=-0.0001000,
# (visualize)                              our_label='BranchyLeNet', orig_label='LeNet', xlabel='Runtime (ms)',
# (visualize)                              title='LeNet GPU', output_path='_figs/lenet_gpu.pdf')



#CPU
# (CPU) branchyNet.to_cpu()
# (CPU) c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
# (CPU)                                                      batchsize=TEST_BATCHSIZE, verbose=True)
# c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, g_ts, inc_amt=0.01,
#                                                      batchsize=TEST_BATCHSIZE, prescreen=False, verbose=True)
#convert to ms
# (CPU) c_diffs *= 1000.



# (visualize) visualize.plot_line_tradeoff(c_accs, c_diffs, c_ts, c_exits, c_baseacc, c_basediff, all_samples=False, inc_amt=-0.0001000,
# (visualize)                              our_label='BranchyLeNet', orig_label='LeNet', xlabel='Runtime (ms)',
# (visualize)                              title='LeNet CPU', output_path='_figs/lenet_cpu.pdf')



#Compute table results
# (CPU) utils.branchy_table_results(c_baseacc, c_basediff, g_basediff, c_accs, c_diffs, g_accs, g_diffs, inc_amt=0.000,
# (CPU)                           network='LeNet')



# (INFERENCE) print '7. GPU Results:'
# (INFERENCE) utils.branchy_table_results('LeNet-MNIST', g_baseacc, g_basediff, g_accs, g_diffs, g_exits, g_ts)

# (INFERENCE) with open("_models/lenet_mnist_gpu_results (inference).pkl", "w") as f:
# (INFERENCE)     dill.dump({'accs': g_accs, 'rt': g_diffs, 'exits': g_exits, 'ts': g_ts, 'baseacc': g_baseacc, 'basediff': g_basediff}, f)
# (CPU) with open("_models/lenet_mnist_cpu_results.pkl", "w") as f:
# (CPU)     dill.dump({'accs': c_accs, 'rt': c_diffs, 'exits': c_exits, 'ts': c_ts, 'baseacc': c_baseacc, 'basediff': c_basediff}, f)
