from Experiments import *
from utils import *
import time

if __name__ == '__main__':
    starttime = time.time()

    print('with analysis shallow...')
    mnist_training_analysis([128, 64], 'results_mnist/')
    print("Done")

    print('That took {} seconds'.format(time.time() - starttime))