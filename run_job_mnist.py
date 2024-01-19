from Experiments import *
from utils import *
import time


if __name__ == '__main__':
    archs = [[128, 64], [512, 64], [512, 128, 64, 32, 16], [512, 256, 128, 64, 32, 16, 12], [10, 16, 32, 128, 256, 128, 64, 32, 16, 10, 784], [10, 512, 256, 128, 64, 32, 16, 16, 16], [10, 10, 10, 10, 10, 10, 10, 10, 10] , [512, 256, 256, 256, 256, 256, 256, 128, 64, 32, 16]]
    starttime = time.time()

    for arch in archs:
        mnist_training_analysis(arch, epochs=20, pre_path='results_mnist_diff_models/')
        print("Done")
        print('That took {} seconds'.format(time.time() - starttime))