from Experiments import *
from utils import *
from tqdm import tqdm
import multiprocessing 
import time
import os
from functools import partial
import argparse
import pandas as pd

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

pp = 'results_new'

def run_function(arch_set, dest='results_new/tests'):
    pool = multiprocessing.Pool(processes=50)
    prod_x=partial(one_random_experiment, exps=50, num=10000, one=False, pre_path=dest, normal_dist=normal_dist, loc=0, scale=scale)
    result_list = pool.map(prod_x, arch_set)

    p_path = dest
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))

    animate_histogram(result_list, ['size:' + str(arch_set[i][0]) for i in range(len(result_list))],  pre_path=p_path)
    pool.close()

def regul():
    print('with analysis shallow...')
    run_function(archs_1, dest='{}/width_analysis/shallow/'.format(pp))
    print("Done")

    print('with analysis deep...')
    run_function(archs_2, dest='{}/width_analysis/deep/'.format(pp))
    print("Done")

    print('with analysis deeper...')
    run_function(archs_3, dest='{}/width_analysis/deeper/'.format(pp))
    print("Done")

if __name__ == '__main__':

    starttime = time.time()
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--normal_dist', metavar='normal_dist', required=True,
                        help='', type=bool)
    parser.add_argument('--scale', metavar='scale', required=True,
                        help='', type=int)
    args = parser.parse_args()

    scale = args.scale
    normal_dist = args.normal_dist
    regul()
    
    print('That took {} seconds'.format(time.time() - starttime))