from Experiments import *
from utils import *
import time


if __name__ == '__main__':

    starttime = time.time()

    # arch = (784, [128, 128, 128, 64, 64, 64, 32, 16, 8, 3])
    # mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='sept_result_mnist_three_class', three_class=True, odd_even=False)

    # arch = (32*32*3, [512, 256, 256, 256, 128, 128, 128, 128, 64, 64, 32, 32, 32, 10])
    # cifar10_training_analysis_hook_engine(3, archirecture=arch, pre_path='cifar10')
    
    arch = (32*32*3, [511, 255, 127, 127, 127, 97, 64, 64, 32, 32, 17, 10])
    three_class_cifar10_training_analysis_spike_loss(0, archirecture=arch, pre_path='three_class_cifar10_with_spike_loss_target', epochs=200)
    

    # arch = (784, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
    # mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='sept_result_mnist_zerobiased', bias=0)
    
    # mnist_training_analysis_hook_engine(6, archirecture=arch, pre_path='sept_result_mnist_odd_evens', three_class=True, odd_even=True)

