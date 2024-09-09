from Experiments import *
from utils import *
import time


if __name__ == '__main__':

    starttime = time.time()

    arch = (784, [256, 128, 128, 128, 128, 64, 64, 64, 64, 64, 32, 32, 32, 32, 10])

    # mnist_training_analysis_hook_engine(0, pre_path='sept_result_mnist')
    # mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='sept_result_mnist_three_class', three_class=True)
    mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='sept_result_mnist_odd_evens', three_class=False, odd_even=True)

