from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda


# Define Network

from networks import lenet_mnist

branchyNet = lenet_mnist.get_network()
# (GPU) branchyNet.to_gpu()
branchyNet.training()


# Import Data

from datasets import mnist
x_train, y_train, x_test, y_test = mnist.get_data()


# Settings

TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 1
TRAIN_NUM_EPOCHS = 50


# Train Main Network


main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)


# Train BranchyNet


TRAIN_NUM_EPOCHS = 100
branch_loss, branch_acc, branch_time = utils.train(branchyNet, x_train, y_train, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)

#set network to inference mode
#branchyNet.testing()


# Visualizing Network Training

# (Visualizing) visualize.plot_layers(main_loss, xlabel='Epochs', ylabel='Training Loss')
# (Visualizing) visualize.plot_layers(main_acc, xlabel='Epochs', ylabel='Training Accuracy')



# (Visualizing) visualize.plot_layers(zip(*branch_loss), xlabel='Epochs', ylabel='Training Loss')
# (Visualizing) visualize.plot_layers(zip(*branch_acc), xlabel='Epochs', ylabel='Training Accuracy')


# Run test suite and visualize


#set network to inference mode
branchyNet.testing()
branchyNet.verbose = False
# (GPU) branchyNet.to_gpu()
# (GPU) g_baseacc, g_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
# (GPU) g_basediff = (g_basediff / float(len(y_test))) * 1000.

branchyNet.to_cpu()
c_baseacc, c_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
c_basediff = (c_basediff / float(len(y_test))) * 1000.


# In[30]:

# Specify thresholds
thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 3., 5., 10.]


# In[20]:

#GPU
# (GPU) branchyNet.to_gpu()
# (GPU) g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
# (GPU)                                                     batchsize=TEST_BATCHSIZE, verbose=True)
# g_ts, g_accs, g_diffs, g_exits = utils.screen_leaky(leakyNet, x_test, y_test, thresholds, inc_amt=-0.1,
#                                                     batchsize=TEST_BATCHSIZE, verbose=True)

#convert to ms
# (GPU) g_diffs *= 1000.



# (Visualizing) visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff, all_samples=False, inc_amt=-0.0001000,
# (Visualizing)                              our_label='BranchyLeNet', orig_label='LeNet', xlabel='Runtime (ms)',
# (Visualizing)                              title='LeNet GPU', output_path='_figs/lenet_gpu.pdf')


# In[32]:

#CPU
branchyNet.to_cpu()
c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
                                                     batchsize=TEST_BATCHSIZE, verbose=True)
# c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, g_ts, inc_amt=0.01,
#                                                      batchsize=TEST_BATCHSIZE, prescreen=False, verbose=True)
#convert to ms
c_diffs *= 1000.


# In[22]:

# (Visualizing) visualize.plot_line_tradeoff(c_accs, c_diffs, c_ts, c_exits, c_baseacc, c_basediff, all_samples=False, inc_amt=-0.0001000,
# (Visualizing)                              our_label='BranchyLeNet', orig_label='LeNet', xlabel='Runtime (ms)',
# (Visualizing)                              title='LeNet CPU', output_path='_figs/lenet_cpu.pdf')


# In[ ]:

#Compute table results
# (GPU) utils.branchy_table_results(c_baseacc, c_basediff, g_basediff, c_accs, c_diffs, g_accs, g_diffs, inc_amt=0.000,
# (GPU)                           network='LeNet')


# Save model/data

# In[40]:

import dill
branchyNet.to_cpu()
with open("_models/lenet_mnist_cpu.bn", "w") as f:
    dill.dump(branchyNet, f)
with open("_models/lenet_mnist_results_cpu.pkl", "w") as f:
    dill.dump({'main_loss': main_loss, 'main_acc': main_acc, 'main_time': main_time, 'branch_loss': branch_loss,
               'branch_acc': branch_acc, 'branch_time': branch_time, 'c_baseacc': c_baseacc, 'c_basediff': c_basediff,
               'c_ts': c_ts, 'c_accs': c_accs, 'c_diffs': c_diffs, 'c_exits': c_exits}, f)
# (GPU) with open("_models/lenet_mnist_gpu_results.pkl", "w") as f:
# (GPU)     dill.dump({'accs': g_accs, 'rt': g_diffs, 'exits': g_exits, 'ts': g_ts, 'baseacc': g_baseacc, 'basediff': g_basediff}, f)

