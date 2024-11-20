from Experiments import *
import multiprocessing 
import time
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

def run_the_whole_thing(archs, normal_dist, scale, constant, parser, pp, projection_analysis_bool):
    pool = multiprocessing.Pool(processes=120)
    prod_x=partial(one_random_experiment, exps=20, num=5000, one=False, 
                   pre_path='{}/width_analysis_{}/'.format(pp, constant), 
                   normal_dist=parser.normal_dist, loc=0, scale=parser.scale, 
                   exp_type=parser.exp_type, projection_analysis_bool=projection_analysis_bool)
    result_list = pool.map(prod_x, archs)

    p_path = '{}/width_analysis_{}/'.format(pp, constant)
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))

    with open(p_path + 'activated_results.pkl', 'wb') as f:
        pickle.dump(result_list, f)
    
    animate_histogram(result_list, 'hidden Layers',  pre_path=p_path)

    pool.close()

if __name__ == '__main__':
    def regul(archs, constant, parser, pp):
        run_the_whole_thing(archs, args.normal_dist, args.scale, constant, parser, pp, args.projection_analysis)
        print("done")
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--normal_dist', metavar='normal_dist', required=True,
                        help='', type=bool)
    parser.add_argument('--scale', metavar='scale', required=True,
                        help='', type=int)
    parser.add_argument('--exp_type', metavar='exp_type', required=False,
                    help='', type=str)
    parser.add_argument('--constant', metavar='constant', required=True,
                help='', type=int)
    parser.add_argument('--projection_analysis', metavar='projection_analysis', required=True,
        help='', type=str2bool)
    args = parser.parse_args()

    constant_depth = args.constant
    init_width = min(constant_depth, 3)
    counter_number = min(4 * constant_depth, 120)
    step = 1
    archs_1 = [(init_width + i * step, [init_width + i * step for _ in range(constant_depth)]) for i in range(counter_number)]
    starttime = time.time()
    pp = 'results_find_starting_point/constant_{}'.format(str(constant_depth))
    regul(archs_1, constant_depth, args, pp)
    print('That took {} seconds'.format(time.time() - starttime))