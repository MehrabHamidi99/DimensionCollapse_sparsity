from Experiments import *
import multiprocessing 
import time
import os
from functools import partial
import pickle
import argparse


constant = 5
archs1 = [(constant, [constant for _ in range(i)]) for i in range(1, 120, 1)]

# constant = 100
# archs2 = [(constant, [constant for _ in range(i)]) for i in range(1, 112, 10)]

pool = multiprocessing.Pool(processes=100)

pp = 'results_find_starting_point/constant_{}'.format(str(constant))

def run_the_whole_thing(archs, normal_dist, scale, constant, parser):
    pool = multiprocessing.Pool(processes=120)
    prod_x=partial(one_random_experiment, exps=50, num=10000, one=False, pre_path='{}/depth_analysis_{}/'.format(pp, constant), normal_dist=parser.normal_dist, loc=0, scale=parser.scale, exp_type=parser.exp_type)
    result_list = pool.map(prod_x, archs)

    p_path = '{}/depth_analysis_{}/'.format(pp, constant)
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))

    with open(p_path + 'activated_results.pkl', 'wb') as f:
        pickle.dump(result_list, f)
    
    animate_histogram(result_list, 'hidden Layers',  pre_path=p_path)

    # pool.close()



if __name__ == '__main__':
    def regul(archs, constant, parser):
        run_the_whole_thing(archs, args.normal_dist, args.scale, constant, parser)
        print("done")
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--normal_dist', metavar='normal_dist', required=True,
                        help='', type=bool)
    parser.add_argument('--scale', metavar='scale', required=True,
                        help='', type=int)
    parser.add_argument('--exp_type', metavar='exp_type', required=True,
                    help='', type=str)
    args = parser.parse_args()
    starttime = time.time()
    # prod1=partial(regul)
    # pool.map(regul, [(archs1, 15), (archs2, 100)])
    regul(archs1, constant, args)
    # regul(archs2, 100)
    pool.close()

    print('That took {} seconds'.format(time.time() - starttime))