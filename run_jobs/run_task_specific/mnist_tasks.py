import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/../..')))
import argparse
import time


from Training.task_specific.mnist_task import *

if __name__ == '__main__':

    starttime = time.time()

    parser_arg = argparse.ArgumentParser(description='')
    parser_arg.add_argument('--training_mode', metavar='training_mode', required=True,
                        help='', type=str)
    
    parser_arg.add_argument('--try_num', metavar='try_num', required=True, type=int)

    parser_arg.add_argument('--bias', metavar='bias', required=False, type=float, default=1e-4)

    parser = vars(parser_arg.parse_args())

    try_num = int(parser['try_num'])
    bias = float(parser['bias'])

    training_mode = parser['training_mode']

    saved_path = f'january_res_png/mnist/{training_mode}/bias_{bias}'


    if parser['training_mode'] == 'normal':
        arch = (784, [256, 128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
        mnist_training_analysis_hook_engine(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias)
    
    if parser['training_mode'] == 'constant_width':
        arch = (784, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 10])
        mnist_training_analysis_hook_engine(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias)

    if parser['training_mode'] == 'normal_three_class':
        arch = (784, [256, 128, 128, 128, 64, 64, 64, 64, 64, 32, 10, 3])
        mnist_training_analysis_hook_engine(try_num, archirecture=arch, pre_path=saved_path, epochs=500, three_class=True, bias=bias)
    
    if parser['training_mode'] == 'normal_two_class':
        arch = (784, [256, 128, 128, 128, 64, 64, 64, 64, 32, 10, 7, 2])
        mnist_training_analysis_hook_engine(try_num, archirecture=arch, pre_path=saved_path, epochs=500, odd_even=True, bias=bias)
    
    #####################

    if parser['training_mode'] == 'spike_loss':
        arch = (784, [256, 128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
        mnist_training_analysis_spike_loss(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias)
    
    if parser['training_mode'] == 'spike_loss_three_class':
        arch = (784, [256, 128, 128, 128, 64, 64, 64, 64, 64, 32, 10, 3])
        mnist_training_analysis_spike_loss(try_num, archirecture=arch, pre_path=saved_path, epochs=500, three_class=True, bias=bias)
    
    if parser['training_mode'] == 'spike_loss_two_class':
        arch = (784, [256, 128, 128, 128, 64, 64, 64, 64, 32, 10, 7, 2])
        mnist_training_analysis_spike_loss(try_num, archirecture=arch, pre_path=saved_path, epochs=500, odd_even=True, bias=bias)
    
    if parser['training_mode'] == 'vit':
        arch = (784, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 7, 2])
        viT_mnist(try_num, pre_path=saved_path, epochs=500, bias=bias)
    
    if parser['training_mode'] == 'debug':
        arch = (784, [256, 128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
        mnist_training_analysis_spike_loss(try_num, archirecture=arch, pre_path=saved_path, epochs=500, bias=bias, debug=True)
    

