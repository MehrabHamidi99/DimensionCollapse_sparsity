from Experiments import *
import multiprocessing 
import time
import os
from functools import partial
import pickle
import argparse


SCALE = -1
LOC = -1
NORMAL = True
BIAS = 0
EXP_TYPE = 'normal'
EXP = 30
NUM=10000

PP = 'new_results_2d_may'



def run_the_whole_thing(tmp_input):
    archs, loc, scale, exp_type_this, bias = tmp_input
    one_random_experiment(architecture=archs,
                          exps=EXP,
                          num=NUM,
                          one=False,
                          loc=loc,
                          scale=scale,
                          pre_path='{}/'.format(PP),
                          normal_dist=NORMAL,
                          exp_type=exp_type_this,
                          projection_analysis_bool=True, 
                          bias=bias
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
    pool.map(prod_x, zip(archs_all, biasses, scales, locs, exp_all))
    
    pool.close()
    pool.join()