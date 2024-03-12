from Experiments import *
import multiprocessing 
import time
import os
from functools import partial
import pickle
import argparse
from utils import *

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
    pool = multiprocessing.Pool(processes=40)
    pre_path_here = '{}/depth_analysis_{}/'.format(pp, constant)
    prod_x=partial(one_random_experiment, exps=20, num=5000, one=False, pre_path=pre_path_here, 
                   normal_dist=parser.normal_dist, loc=0, scale=parser.scale, exp_type=parser.exp_type, 
                   projection_analysis_bool=projection_analysis_bool)
    result_list = pool.map(prod_x, archs)

    p_path = file_name_handling(which=None, architecture=None, num=num, exps=exps, pre_path=pre_path_here, normal_dist=normal_dist, loc=0, scale=scale, bias=1e-4, exp_type='normal', model_type='mlp', return_pre_path=True)

    # p_path = '{}/depth_analysis_{}/'.format(pp, constant)
    # if normal_dist:
    #     p_path +=  'normal_std{}/'.format(str(scale))

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
                        help='', type=str2bool)
    parser.add_argument('--scale', metavar='scale', required=True,
                        help='', type=int)
    parser.add_argument('--exp_type', metavar='exp_type', required=False,
                    help='', type=str)
    parser.add_argument('--constant', metavar='constant', required=True,
                help='', type=int)
    parser.add_argument('--projection_analysis', metavar='projection_analysis', required=True,
            help='', type=str2bool)
    args = parser.parse_args()
    constant = args.constant
    print(args)
    # archs1 = [(constant, [constant for _ in range(i)]) for i in range(1, min(constant * 10, 120), 1)]
    archs1 = [(constant, [constant for _ in range(i)]) for i in [10, 20, 50, 100,]]

    starttime = time.time()
    pp = 'results_find_starting_point/constant_{}'.format(str(constant))
    # prod1=partial(regul)
    # pool.map(regul, [(archs1, 15), (archs2, 100)])
    regul(archs1, constant, args, pp)
    # regul(archs2, 100)

    print('That took {} seconds'.format(time.time() - starttime))