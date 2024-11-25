import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/../..')))
import argparse
import time

from Training.task_specific.cifar10_task import *

if __name__ == '__main__':

    starttime = time.time()

    parser_arg = argparse.ArgumentParser(description='')
    parser_arg.add_argument('--training_mode', metavar='training_mode', required=True,
                        help='', type=str)
    
    parser_arg.add_argument('--try_num', metavar='try_num', required=True, type=int)

    parser_arg.add_argument('--bias', metavar='bias', required=False, type=float)

    parser = vars(parser_arg.parse_args())

    try_num = int(parser['try_num'])
    bias = float(parser['bias'])

    training_mode = parser['training_mode']

    saved_path = f'november_res/cifar10/{training_mode}/bias_{bias}'
    
    if parser['training_mode'] == 'normal':
        arch = (32*32*3, [1024, 784, 784, 512, 256, 256, 128, 128, 64, 64, 32, 32, 10, 10])
        cifar10_training_analysis_hook_engine(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias)
    
    if parser['training_mode'] == 'normal_three_class':
        arch = (32*32*3, [1024, 512, 256, 256, 128, 128, 64, 64, 32, 32, 10, 3])
        three_class_cifar10_training_analysis_hook_engine(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias)

    #####################

    if parser['training_mode'] == 'spike_loss':
        arch = (32*32*3, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
        cifar10_training_analysis_spike_loss(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias)
    
    if parser['training_mode'] == 'spike_loss_three_class':
        arch = (32*32*3, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10, 3])
        three_class_cifar10_training_analysis_spike_loss(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias)
   
    ##################


    if parser['training_mode'] == 'mlp_mixer':
        cifar10_training_analysis_hook_engine_resnetMLP(try_num, pre_path=saved_path, epochs=500, bias=bias)

    ##################

    if parser['training_mode'] == 'res_net':
        arch = (32*32*3, [1024, 512, 256, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 64, 32, 10])
        cifar10_training_analysis_spike_loss_resnet(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias)
    
    
