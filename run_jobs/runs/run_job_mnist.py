from Experiments import *
from utils import *
import time


if __name__ == '__main__':

    starttime = time.time()

    # arch = (784, [128, 128, 128, 64, 64, 64, 32, 16, 8, 3])
    # mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='sept_result_mnist_three_class', three_class=True, odd_even=False)

    # arch = (784, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
    # mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='sept_result_mnist')
    
    arch = (784, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
    mnist_training_analysis_spike_loss(1, archirecture=arch, pre_path='november_res/spike_loss_results_mnist_neural_collapse', epochs=500)
    
    # mnist_training_analysis_hook_engine(6, archirecture=arch, pre_path='sept_result_mnist_odd_evens', three_class=True, odd_even=True)

