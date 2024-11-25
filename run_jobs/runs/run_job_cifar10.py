import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/..')))


from Training.Experiments import *
from utils import *
import time


if __name__ == '__main__':

    starttime = time.time()

    # arch = (784, [128, 128, 128, 64, 64, 64, 32, 16, 8, 3])
    # mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='sept_result_mnist_three_class', three_class=True, odd_even=False)

    # arch = (32*32*3, [512, 256, 256, 256, 128, 128, 128, 128, 64, 64, 32, 32, 32, 10])
    # cifar10_training_analysis_hook_engine(3, archirecture=arch, pre_path='cifar10')
    
    arch = (32*32*3, [1001, 1001, 511, 511, 511, 101, 255, 255, 255, 255, 101, 127, 127, 127, 127, 101, 97, 97, 61, 61, 101, 31, 31, 31, 31, 101, 17, 17, 17, 17, 101, 7, 7, 7, 10])
    cifar10_training_analysis_spike_loss(1, archirecture=arch, pre_path='cifar10_with_spike_loss_neural_collapse_long', epochs=500)

    # arch = (784, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
    # mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='sept_result_mnist_zerobiased', bias=0)
    
    # mnist_training_analysis_hook_engine(6, archirecture=arch, pre_path='sept_result_mnist_odd_evens', three_class=True, odd_even=True)

