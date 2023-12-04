from Experiments import *
from tqdm import tqdm
import multiprocessing 
import time
import os
from functools import partial
import pickle

constant = 15
archs1 = [(constant, [constant for _ in range(i)]) for i in range(1, 112, 5)]

constant = 100
archs2 = [(constant, [constant for _ in range(i)]) for i in range(1, 112, 5)]

def run_the_whole_thing(archs, normal_dist, scale, constant):
    pool = multiprocessing.Pool(processes=50)
    prod_x=partial(one_random_experiment, exps=50, num=10000, one=False, pre_path='results_init/depth_analysis_{}/'.format(constant), normal_dist=normal_dist, loc=0, scale=scale)
    result_list = pool.map(prod_x, archs)

    p_path = 'results_init/depth_analysis_{}/'.format(constant)
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))

    with open(p_path + 'activated_results.pkl', 'wb') as f:
        pickle.dump(result_list, f)
    
    animate_histogram(result_list, 'hidden Layers',  pre_path=p_path)

    pool.close()

def regul(archs, constant):
    scale = 1
    normal_dist = True
    run_the_whole_thing(archs, normal_dist, scale, constant)

    scale = 10
    normal_dist = True
    run_the_whole_thing(archs, normal_dist, scale, constant)

    scale = 1
    normal_dist = False
    run_the_whole_thing(archs, normal_dist, scale, constant)

if __name__ == '__main__':

    starttime = time.time()

    regul(archs1, 15)
    regul(archs2, 100)

    print('That took {} seconds'.format(time.time() - starttime))