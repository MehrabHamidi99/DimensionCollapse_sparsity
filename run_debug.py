import os

from Experiments import *
from tqdm import tqdm
from Models import *
from utils import *
from DataGenerator import *
from DataLoader import *

import time

scale = 10
loc = 0
normal_dist = True
archs = [
    # (2, [10, 20, 30]),
    # (2, [2, 2, 2]),
    # (2, [2, 2, 2, 2]),
    # (2, [2, 2, 2, 2, 2]),
    # (3, [3, 3, 3, 3, 3]),

    
    (2, [100 for _ in range(10)]),
    # (2, [16, 64, 256, 256, 256, 256]),
    # (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
    # (2, [4])
    # (2, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
]


data_properties = {
    'exp_type':'fixed',
    'normal_dist': normal_dist, 
    'loc': loc, 
    'scale': scale
}


s_t = time.time()
results = [random_experiment_hook_engine(i, exps=10, num=1000, pre_path='test_results/', data_properties=data_properties, model_type='mlp') for i in tqdm(archs)]
print(time.time() - s_t)