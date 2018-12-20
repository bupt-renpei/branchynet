import cPickle as pickle
f = open('lenet_mnist_gpu_results.pkl')
#f = open('alexnet_cifar10_gpu_results.pkl')
#f = open('resnet_cifar10_gpu_results.pkl')
info = pickle.load(f)
print info