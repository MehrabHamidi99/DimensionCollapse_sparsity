from Experiments import *
from tqdm import tqdm
import multiprocessing 
import time
import os
from functools import partial

constant_width = 50
constant_depth = 4
archs = [(constant_width + i * 10, [constant_width + i * 10 for _ in range(constant_depth)]) for i in range(10)]

constant_depth = 10
archs2 = [(constant_width + i * 10, [constant_width + i * 10 for _ in range(constant_depth)]) for i in range(10)]


# archs = [
#     (100, [100]),
#     (100, [100, 100]),
#     (100, [100, 100, 100]),
#     (100, [100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
#     (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),

# ]

if __name__ == '__main__':
    scale = 1
    normal_dist = True

    starttime = time.time()
    pool = multiprocessing.Pool(processes=40)
    prod_x=partial(one_random_experiment, exps=50, num=10000, one=False, pre_path='results/width_analysis/', normal_dist=normal_dist, loc=0, scale=scale)
    result_list = pool.map(prod_x, archs)

    p_path = 'results/width_analysis/'
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))

    animate_histogram(result_list, 'hidden Layers',  pre_path=p_path)

    pool.close()
    print("second batch")

    pool = multiprocessing.Pool(processes=40)
    prod_x=partial(one_random_experiment, exps=50, num=10000, one=False, pre_path='results/width_analysis1/', normal_dist=normal_dist, loc=0, scale=scale)
    result_list = pool.map(prod_x, archs2)

    p_path = 'results/width_analysis1/'
    if normal_dist:
        p_path +=  'normal_std{}/'.format(str(scale))

    animate_histogram(result_list, 'hidden Layers',  pre_path=p_path)

    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))