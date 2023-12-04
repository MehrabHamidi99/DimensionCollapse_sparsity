from Experiments import *
from utils import *
from tqdm import tqdm
import multiprocessing 
import time
import os
from functools import partial

archs = [
    (2, [5, 5, 5, 5, 5, 5, 5, 5, 5, 1]),
    (800, [400, 200, 100, 50, 20, 10, 1]),
    (2, [400, 200, 100, 50, 20, 10, 1]),
    (200, [400, 200, 100, 50, 50, 50, 50, 50, 50, 50, 50, 50, 20, 10, 1]),
    (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1]),
    (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 1]),
    (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
    (2, [4, 8, 16, 32, 64, 128, 512, 1024, 2048, 1]),
    (2, [4, 8, 16, 32, 64, 128, 512, 1024, 2048, 1024, 512, 256, 128, 64, 32, 16, 8 , 4, 2, 1]),
    (784, [512, 256, 128, 64, 32, 16, 8, 4, 2, 1]),
    (100, [2, 3, 5, 1])
]

def run_function(arch_set, dest='results/tests'):
    pool = multiprocessing.Pool(processes=40)
    prod_x = partial(before_after_training_experiment, epochs=50, num=10000, pre_path=dest, normal_dist=normal_dist, loc=0, scale=scale)

    result_list = pool.map(prod_x, arch_set)

    p_path = dest
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))

    animate_histogram(result_list, 'different architectures after training',  pre_path=p_path)
    pool.close()

if __name__ == '__main__':
    scale = 1
    normal_dist = True
    starttime = time.time()

    print('with analysis shallow...')
    run_function(archs, 'results_init/training_analysis/')
    print("Done")

    print('That took {} seconds'.format(time.time() - starttime))