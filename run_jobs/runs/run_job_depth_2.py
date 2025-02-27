from Experiments import *
import multiprocessing 
import time
import os
from functools import partial
import pickle
import argparse
from tqdm import tqdm

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run_the_whole_thing(archs, normal_dist, scale, constant, parser, pp, projection_analysis_bool, bias):
    pool = multiprocessing.Pool(processes=40)
    prod_x=partial(one_random_experiment, exps=20, num=10000, one=False, pre_path='{}/'.format(pp), 
                   normal_dist=parser.normal_dist, loc=0, scale=parser.scale, exp_type=parser.exp_type, 
                   projection_analysis_bool=projection_analysis_bool, bias=bias)
    result_list = pool.map(prod_x, archs)

    p_path = '{}/'.format(pp)
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))
    p_path += 'bias_{}/'.format(str(bias))

    with open(p_path + 'activated_results.pkl', 'wb') as f:
        pickle.dump(result_list, f)
    titles = ['hidden Layers' + str(len(arch[1])) for arch in archs]
    animate_histogram(result_list, title=titles,  pre_path=p_path)

    pool.close()


if __name__ == '__main__':
    def regul(archs, constant, parser, pp, bias):
        run_the_whole_thing(archs, args.normal_dist, args.scale, constant, parser, pp, args.projection_analysis, bias)
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
    # constant = args.constant
    constants = [5, 10, 20, 60, 100]
    biasses = [0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, -1e-1, -1e-2, -1e-3, -1e-4, -1e-5]
    for constant in tqdm(constants):
        for bias in tqdm(biasses):
            archs1 = [(2, [constant for _ in range(i)]) for i in range(1, min(constant * 10, 120), 10)] + [(constant, [constant for _ in range(i)]) for i in range(1, min(constant * 10, 120), 10)] 
            starttime = time.time()
            pp = 'results_2d_biasses/constant_{}'.format(str(constant))
            # prod1=partial(regul)
            # pool.map(regul, [(archs1, 15), (archs2, 100)])
            regul(archs1, constant, args, pp, bias)
            # regul(archs2, 100)

            print('That took {} seconds'.format(time.time() - starttime))