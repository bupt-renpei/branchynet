from branchynet.net import BranchyNet
from branchynet.links import *
import chainer.functions as F
import chainer.links as L
from branchynet import utils, visualize
from chainer import cuda


# Define Network

# In[3]:

# rprint 'chainer.__version__ : ', chainer.__version__

from networks import lenet_mnist

print '1. lenet_mnist.get_network()'

branchyNet = lenet_mnist.get_network()
branchyNet.to_gpu()
branchyNet.training()


# Import Data

# In[4]:

print '2. mnist.get_data()'

from datasets import mnist
x_train, y_train, x_test, y_test = mnist.get_data()


# Settings

# In[5]:

TRAIN_BATCHSIZE = 512
TEST_BATCHSIZE = 1
TRAIN_NUM_EPOCHS = 50


# Train Main Network

# In[6]:

main_loss, main_acc, main_time = utils.train(branchyNet, x_train, y_train, main=True, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)

print '3. main_acc : ', main_acc

# Train BranchyNet

# In[7]:

TRAIN_NUM_EPOCHS = 100
branch_loss, branch_acc, branch_time = utils.train(branchyNet, x_train, y_train, batchsize=TRAIN_BATCHSIZE,
                                             num_epoch=TRAIN_NUM_EPOCHS)

print '4. branch_acc : ', branch_acc

#set network to inference mode
# (redundance) branchyNet.testing()


# Visualizing Network Training

# In[8]:

# (visualize) visualize.plot_layers(main_loss, xlabel='Epochs', ylabel='Training Loss')
# (visualize) visualize.plot_layers(main_acc, xlabel='Epochs', ylabel='Training Accuracy')


# In[9]:

# (visualize) visualize.plot_layers(zip(*branch_loss), xlabel='Epochs', ylabel='Training Loss')
# (visualize) visualize.plot_layers(zip(*branch_acc), xlabel='Epochs', ylabel='Training Accuracy')


# Run test suite and visualize

# In[11]:

#set network to inference mode

print '5. Set network to inference mode'

print '5.1 branchyNet.testing()'
branchyNet.testing()
branchyNet.verbose = False

branchyNet.to_gpu()
g_baseacc, g_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
g_basediff = (g_basediff / float(len(y_test))) * 1000.

print '  g_baseacc : ', g_baseacc

#branchyNet.to_cpu()
# (CPU) c_baseacc, c_basediff, _, _ = utils.test(branchyNet,x_test,y_test,main=True,batchsize=TEST_BATCHSIZE)
# (CPU) c_basediff = (c_basediff / float(len(y_test))) * 1000.


# In[30]:

# Specify thresholds
thresholds = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1., 2., 3., 5., 10.]


# In[20]:

#GPU
branchyNet.to_gpu()
g_ts, g_accs, g_diffs, g_exits = utils.screen_branchy(branchyNet, x_test, y_test, thresholds, batchsize=TEST_BATCHSIZE, verbose=True)

print '  g_accs : ', g_accs

# g_ts, g_accs, g_diffs, g_exits = utils.screen_leaky(leakyNet, x_test, y_test, thresholds, inc_amt=-0.1,
#                                                     batchsize=TEST_BATCHSIZE, verbose=True)

#convert to ms
g_diffs *= 1000.


# In[ ]:

# (visualize) visualize.plot_line_tradeoff(g_accs, g_diffs, g_ts, g_exits, g_baseacc, g_basediff, all_samples=False, inc_amt=-0.0001000,
# (visualize)                              our_label='BranchyLeNet', orig_label='LeNet', xlabel='Runtime (ms)',
# (visualize)                              title='LeNet GPU', output_path='_figs/lenet_gpu.pdf')


# In[32]:

#CPU
# (CPU) branchyNet.to_cpu()
# (CPU) c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, thresholds,
# (CPU)                                                      batchsize=TEST_BATCHSIZE, verbose=True)
# c_ts, c_accs, c_diffs, c_exits  = utils.screen_branchy(branchyNet, x_test, y_test, g_ts, inc_amt=0.01,
#                                                      batchsize=TEST_BATCHSIZE, prescreen=False, verbose=True)
#convert to ms
# (CPU) c_diffs *= 1000.


# In[22]:

# (visualize) visualize.plot_line_tradeoff(c_accs, c_diffs, c_ts, c_exits, c_baseacc, c_basediff, all_samples=False, inc_amt=-0.0001000,
# (visualize)                              our_label='BranchyLeNet', orig_label='LeNet', xlabel='Runtime (ms)',
# (visualize)                              title='LeNet CPU', output_path='_figs/lenet_cpu.pdf')


# In[ ]:

#Compute table results
# (CPU) utils.branchy_table_results(c_baseacc, c_basediff, g_basediff, c_accs, c_diffs, g_accs, g_diffs, inc_amt=0.000,
# (CPU)                           network='LeNet')

print '6. GPU Results:'
utils.branchy_table_results('LeNet-MNIST', g_baseacc, g_basediff, g_accs, g_diffs, g_exits, g_ts)


# Save model/data

# In[40]:

print '7. Save mode'
import dill
# (CPU) branchyNet.to_cpu()
branchyNet.to_gpu()
with open("_models/lenet_mnist.bn", "w") as f:
    dill.dump(branchyNet, f)
with open("_models/lenet_mnist_gpu_results.pkl", "w") as f:
    dill.dump({'accs': g_accs, 'rt': g_diffs, 'exits': g_exits, 'ts': g_ts, 'baseacc': g_baseacc, 'basediff': g_basediff}, f)
# (CPU) with open("_models/lenet_mnist_cpu_results.pkl", "w") as f:
# (CPU)     dill.dump({'accs': c_accs, 'rt': c_diffs, 'exits': c_exits, 'ts': c_ts, 'baseacc': c_baseacc, 'basediff': c_basediff}, f)
