from Experiments import *
from utils import *
from tqdm import tqdm
import multiprocessing 
import time
import os
from functools import partial

counter_number = 30
step = 5

constant_width = 5
constant_depth = 3
archs_1 = [(constant_width + i * step, [constant_width + i * step for _ in range(constant_depth)]) for i in range(counter_number)]


constant_width = 5
constant_depth = 12
archs_2 = [(constant_width + i * step, [constant_width + i * step for _ in range(constant_depth)]) for i in range(counter_number)]


constant_width = 5
constant_depth = 64
archs_3 = [(constant_width + i * step, [constant_width + i * step for _ in range(constant_depth)]) for i in range(counter_number)]


def run_function(arch_set, dest='results/tests'):
    pool = multiprocessing.Pool(processes=40)
    prod_x=partial(one_random_experiment, exps=50, num=10000, one=False, pre_path=dest, normal_dist=normal_dist, loc=0, scale=scale)
    result_list = pool.map(prod_x, arch_set)

    p_path = dest
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))

    animate_histogram(result_list, 'hidden Layers',  pre_path=p_path)
    pool.close()

if __name__ == '__main__':
    scale = 1
    normal_dist = True

    starttime = time.time()

    print('with analysis shallow...')
    run_function(archs_1, 'results/width_analysis/shallow/')
    print("Done")

    print('with analysis deep...')
    run_function(archs_2, 'results/width_analysis/deep/')
    print("Done")

    print('with analysis deeper...')
    run_function(archs_3, 'results/width_analysis/deeper/')
    print("Done")

    print('That took {} seconds'.format(time.time() - starttime))