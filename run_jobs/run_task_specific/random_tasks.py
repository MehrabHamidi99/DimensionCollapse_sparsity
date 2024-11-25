from Training.task_specifc import random_experiment_task
import time

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/../..')))
import multiprocessing 
import time
import os
from functools import partial
import pickle
import argparse

import itertools


if __name__ == '__main__':

    starttime = time.time()

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--mode', metavar='mode', required=True,
                        help='', type=str2bool, default='custome', choices=['custome', 'depth', 'width', 'single_layer'])
    
    parser.add_argument('--exp_type', metavar='exp_type', required=True,
                        help='', type=str2bool, choices=['plane', 'normal', 'line', 'fixed', 'line_d'])
    
    parser.add_argument('--try_num', metavar='try_num', required=True, type=int)

    parser.add_argument('--bias', metavar='bias', required=False, type=float)
    parser.add_argument('--new_model', metavar='new_model', required=False, type=bool)
    parser.add_argument('--scale', metavar='scale', required=False, help='', type=int, default=1)
    parser.add_argument('--loc', metavar='loc', required=False, help='', type=int, default=0)

    parser.add_argument('--constant', metavar='contant', required=False, help='', type=int, default=0)

    mode = parser['mode']

    try_num = int(parser['try_num'])
    bias = float(parser['bias'])
    exp_type = parser['exp_type']
    scale = float(parser['scale'])
    loc = float(parser['loc'])
    new_model_each_time = bool(parser['new_model'])

    constant = int(parser['constant'])

    EXP = 30
    NUM=10000

    data_prop = {
        'normal_dist': True, 
        'loc': loc, 
        'scale': scale, 
        'exp_type': exp_type
    }

    pp = f'november_res/rondom_experiment/{mode}'

    elif mode == 'custome':
        archs = [
            (2, [10 for _ in range(5)]),
            (2, [10 for _ in range(20)]),
            (2, [10 for _ in range(60)]),
            
            (2, [30 for _ in range(5)]),
            (2, [30 for _ in range(20)]),
            (2, [30 for _ in range(60)]),
            
            (2, [100 for _ in range(5)]),
            (2, [100 for _ in range(20)]),
            (2, [100 for _ in range(60)]),


            (10, [10 for _ in range(5)]),
            (10, [10 for _ in range(20)]),
            (10, [10 for _ in range(60)]),
            
            (30, [30 for _ in range(5)]),
            (30, [30 for _ in range(20)]),
            (30, [30 for _ in range(60)]),
            
            (100, [100 for _ in range(5)]),
            (100, [100 for _ in range(20)]),
            (100, [100 for _ in range(60)]),


            (100, [100 for _ in range(100)]),

        ]
    if mode == 'depth':
        archs = [(2, [constant for _ in range(i)]) for i in [10, 20, 50, 100,]] + [(constant, [constant for _ in range(i)]) for i in [10, 20, 50, 100]]

   
    elif mode == 'width':
        init_width = 5
        archs = [(init_width + i, [init_width + i for _ in range(constant)]) for i in [0, 5, 10, 25, 55, 65, 85, 95, 105]] + [(2, [init_width + i for _ in range(constant)]) for i in [0, 5, 10, 25, 55, 65, 85, 95, 105]]
    
    elif mode == 'single_layer':
        archs = [(i, [i]) for i in range(2, 65, 10)]
    else:
        raise Exception("Undefined mode!")


    pool = multiprocessing.Pool(processes=40)
    prod_x=partial(random_experiment_hook_engine, 
                    exps=EXP, 
                    num=NUM, 
                    data_properties=data_prop,
                    pre_path='{}/'.format(pp), 
                    bias=bias,
                    model_type='mlp', 
                    new_model_each_time=new_model_each_time
                    )
    pool.map(prod_x, archs)

