from Experiments import *
import multiprocessing 
import time
import os
from functools import partial
import pickle
import argparse

import itertools


SCALE = -1
LOC = -1
NORMAL = True
BIAS = 0
EXP_TYPE = 'normal'
EXP = 30
NUM=10000

PP = 'new_results_2d_june'



def run_the_whole_thing(tmp_input):
    print(tmp_input)
    archs, bias, scale, loc, exp_type_this = tmp_input
    data_prop = {
        'normal_dist': True, 
        'loc': loc, 
        'scale': scale, 
        'exp_type': exp_type_this
    }
    random_experiment_hook_engine(architecture=archs,
                          exps=EXP,
                          num=NUM,
                          data_properties=data_prop,
                          pre_path='{}/'.format(PP),
                          bias=bias,
                          model_type='mlp'
                          )


if __name__ == '__main__':
    
    archs_all = [
        (2, [10 for _ in range(30)]),
        (2, [8 for _ in range(30)]),
        (2, [100 for _ in range(50)]),
        (2, [100 for _ in range(30)]),
        (2, [100 for _ in range(60)]),
        (2, [100 for _ in range(70)]),
        (2, [100 for _ in range(80)]),
        (2, [100 for _ in range(90)]),

        (2, [20 for _ in range(50)]),
        (2, [20 for _ in range(60)]),
        (2, [40 for _ in range(60)]),
        (2, [60 for _ in range(60)]),


        (10, [10 for _ in range(30)]),
        (8, [8 for _ in range(30)]),
        
        (100, [100 for _ in range(50)]),
        (100, [100 for _ in range(30)]),
        (100, [100 for _ in range(60)]),
        (100, [100 for _ in range(90)]),

        (20, [20 for _ in range(50)]),
        (20, [20 for _ in range(70)])
    ]
    biasses = [0, 1e-4]
    scales = [1, 100, 500]
    locs = [0, 50]
    exp_all = ['plane', 'normal', 'line', 'fixed']

    pool = multiprocessing.Pool(processes=40)
    prod_x=partial(run_the_whole_thing)
    pool.map(prod_x, itertools.product(archs_all, biasses, scales, locs, exp_all))
    
    pool.close()
    pool.join()