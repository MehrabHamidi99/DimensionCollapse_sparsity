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


SCALE = -1
LOC = -1
NORMAL = True
BIAS = -1
EXP_TYPE = 'normal'


def run_the_whole_thing(archs, normal_dist, scale, parser, pp, projection_analysis_bool):
    pool = multiprocessing.Pool(processes=40)
    prod_x=partial(one_random_experiment, exps=30, num=5000, one=False, pre_path='{}/'.format(pp), 
                   normal_dist=NORMAL, loc=LOC, scale=SCALE, exp_type=EXP_TYPE, 
                   projection_analysis_bool=projection_analysis_bool, bias=BIAS, model_type='res_net')
    result_list = pool.map(prod_x, archs)
    p_path = '{}'.format(pp)

    p_path = file_name_handling(which=None, architecture=None, num=5000, exps=30, pre_path=p_path, normal_dist=NORMAL, loc=LOC, scale=SCALE, bias=BIAS, exp_type=EXP_TYPE, model_type='res_net', return_pre_path=True)

    # if NORMAL:
    #     p_path += '{}_mean_{}_std{}/'.format(str(EXP_TYPE), str(LOC), str(SCALE))
    # else:
    #     p_path += 'uniform/'
    # p_path += 'bias_{}/'.format(str(bias))

    # with open(p_path + 'activated_results.pkl', 'wb') as f:
    #     pickle.dump(result_list, f)
    # titles = ['hidden Layers' + str(len(arch[1])) for arch in archs]
    # animate_histogram(result_list, title=titles,  pre_path=p_path)

    pool.close()


if __name__ == '__main__':
    def regul(archs, parser, pp):
        # run_the_whole_thing(archs, NORMAL, SCALE, parser, pp, args.projection_analysis)
        run_the_whole_thing(archs, NORMAL, SCALE, parser, pp, True)
        print("done", flush=True)
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--normal_dist', metavar='normal_dist', required=False,
                        help='', type=str2bool)
    parser.add_argument('--scale', metavar='scale', required=False,
                        help='', type=int)
    parser.add_argument('--exp_type', metavar='exp_type', required=False,
                    help='', type=str)
    parser.add_argument('--projection_analysis', metavar='projection_analysis', required=False,
            help='', type=str2bool)
    args = parser.parse_args()
    archs_all = [
        (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]),
        (100, [100 for _ in range(50)]),
        (100, [100 for _ in range(30)]),
        (10, [10 for _ in range(30)]),
        (8, [8 for _ in range(30)]),
        (20, [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]),
        (20, [20 for _ in range(50)]),
    ]
    biasses = [0, 1e-4]
    scales = [1, 100]
    locs = [0]
    pp = 'results_resNet3/'

    for bias in biasses:
        BIAS = bias
        for scale in scales:   
            SCALE = scale
         
            EXP_TYPE = 'normal'
            for loc in locs:
                LOC = loc
                NORMAL = True
                # archs1 = [(constant, [constant for _ in range(i)]) for i in range(1, 30, 1)]
                starttime = time.time()
                # prod1=partial(regul)
                # pool.map(regul, [(archs1, 15), (archs2, 100)])
                # print(arch_this, 'arch this')
                regul(archs_all, args, pp)
                # regul(archs2, 100)

                print('That took {} seconds'.format(time.time() - starttime), flush=True)

            EXP_TYPE = 'fixed'
            LOC = loc
            NORMAL = True
            # archs1 = [(constant, [constant for _ in range(i)]) for i in range(1, 30, 1)]
            starttime = time.time()
            # prod1=partial(regul)
            # pool.map(regul, [(archs1, 15), (archs2, 100)])
            # print(arch_this, 'arch this')
            regul(archs_all, args, pp)
            # regul(archs2, 100)

            print('That took {} seconds'.format(time.time() - starttime), flush=True)


        EXP_TYPE = 'normal'
        NORMAL = False
        # archs1 = [(constant, [constant for _ in range(i)]) for i in range(1, 30, 1)]
        starttime = time.time()
        # prod1=partial(regul)
        # pool.map(regul, [(archs1, 15), (archs2, 100)])
        regul(archs_all, args, pp)
        # regul(archs2, 100)

        print('That took {} seconds'.format(time.time() - starttime), flush=True)