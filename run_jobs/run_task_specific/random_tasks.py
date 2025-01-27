import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/../..')))
import argparse
import time
import multiprocessing 
import os
from functools import partial
import pickle
import argparse
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

from Training.task_specific.random_experiment_task import *


if __name__ == '__main__':

    starttime = time.time()

    parser_arg = argparse.ArgumentParser(description='')

    parser_arg.add_argument('--mode', metavar='mode', required=True,
                        help='', type=str, default='custome', choices=['custome', 'depth', 'width', 'single_layer'])
    
    parser_arg.add_argument('--exp_type', metavar='exp_type', required=True,
                        help='', type=str, choices=['plane', 'normal', 'line', 'fixed', 'line_d'])
    
    parser_arg.add_argument('--try_num', metavar='try_num', required=True, type=int)

    parser_arg.add_argument('--bias', metavar='bias', required=False, type=float)
    parser_arg.add_argument('--new_model', metavar='new_model', required=False, type=str2bool)
    parser_arg.add_argument('--scale', metavar='scale', required=False, help='', type=float, default=1)
    parser_arg.add_argument('--loc', metavar='loc', required=False, help='', type=float, default=0)

    parser_arg.add_argument('--debug', metavar='debug', required=False, help='', type=str2bool, default=False)

    # parser_arg.add_argument('--constant', metavar='contant', required=False, help='', type=int, default=0)

    parser = vars(parser_arg.parse_args())

    print(parser)

    mode = parser['mode']

    try_num = int(parser['try_num'])
    bias = float(parser['bias'])
    exp_type = parser['exp_type']
    scale = float(parser['scale'])
    loc = float(parser['loc'])
    new_model_each_time = bool(parser['new_model'])

    # constant = int(parser['constant'])

    EXP = 30
    NUM=5000

    data_prop = {
        'normal_dist': True, 
        'loc': loc, 
        'scale': scale, 
        'exp_type': exp_type
    }

    pp = f'/network/scratch/m/mehrab.hamidi/januery_res25_scale_fixed_batch_norm/random_experiment/{mode}'

    if mode == 'custome':
        archs = [
            # (2, [10 for _ in range(5)]),
            (2, [10 for _ in range(23)]),
            # (2, [10 for _ in range(63)]),
            
            # (2, [30 for _ in range(5)]),
            (2, [30 for _ in range(23)]),
            (2, [30 for _ in range(63)]),
            
            # (2, [100 for _ in range(5)]),
            (2, [100 for _ in range(23)]),
            # (2, [100 for _ in range(63)]),


            # (10, [10 for _ in range(5)]),
            (10, [10 for _ in range(23)]),
            # (10, [10 for _ in range(63)]),
            
            # (30, [30 for _ in range(5)]),
            (30, [30 for _ in range(23)]),
            (30, [30 for _ in range(63)]),
            
            # (100, [100 for _ in range(5)]),
            (100, [100 for _ in range(23)]),
            # (100, [100 for _ in range(63)]),


            (100, [100 for _ in range(111)]),

        ]
    elif mode == 'depth':
        archs = []
        for constant in [5, 20, 40, 80]:
            archs += [(2, [constant for _ in range(i)]) for i in [11, 23, 53, 111]] + [(constant, [constant for _ in range(i)]) for i in [11, 23, 53, 111]]

    elif mode == 'width':
        archs = []
        init_width = 5
        for constant in [5, 15, 31, 53, 83]:
            archs += [(init_width + i, [init_width + i for _ in range(constant)]) for i in [0, 5, 10, 25, 65, 85, 105]] + [(2, [init_width + i for _ in range(constant)]) for i in [0, 5, 10, 25, 55, 65, 85, 95, 105]]
    
    elif mode == 'single_layer':
        archs = [(i, [i]) for i in range(2, 65, 10)]
    else:
        raise Exception("Undefined mode!")

    if parser['debug']:
        for arch in archs:
            pp += '/debug'
            random_experiment_hook_engine(arch, exps=EXP, 
                                            num=NUM, 
                                            data_properties=data_prop,
                                            pre_path='{}/'.format(pp), 
                                            bias=bias,
                                            model_type='mlp', 
                                            new_model_each_time=new_model_each_time,
                                            new_data_each_time=True)

    else:
        pool = multiprocessing.Pool(processes=min(40, len(archs)))
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

