from Training.task_specifc import kuji_mnist_task
import time

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/../..')))



if __name__ == '__main__':

    starttime = time.time()

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--training_mode', metavar='training_mode', required=True,
                        help='', type=str2bool)
    
    parser.add_argument('--try_num', metavar='try_num', required=True, type=int)

    parser.add_argument('--bias', metavar='bias', required=False, type=float)

    try_num = int(parser['try_num'])
    bias = float(parser['bias'])


    if parser['training_mode'] == 'normal':
        arch = (784, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
        kuji_mnist_training_analysis_hook_engine(try_num, archirecture=arch, pre_path='november_res/kuji_mnist_normal', epochs=500, bias=bias)
    
    #####################

    if parser['training_mode'] == 'spike_loss':
        arch = (784, [128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
        kuji_mnist_training_analysis_spike_loss(try_num, archirecture=arch, pre_path='november_res/kuji_mnist_spike_loss', epochs=500, bias=bias)

