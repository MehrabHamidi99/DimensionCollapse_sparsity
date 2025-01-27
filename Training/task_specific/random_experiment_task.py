from Models.Models import *
from Data.DataGenerator import *
from utils import *
from Training.Training import *
from Models.ForwardPass import *
from Models.Models_normal import *
from Models.FeatureExtractor import *
from Training.Analysis import *
import torch
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from Models.VisionTranformer import VisionTransformer

import wandb
from Models.other_models import MLPMixer


def random_experiment_hook_engine(architecture, exps=50, num=1000, pre_path='', data_properties={'normal_dist': True, 'loc': 0, 'scale': 1, 'exp_type': 'normal'}, bias: float=0.0, model_type='mlp', new_model_each_time=False, new_data_each_time=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    this_path = file_name_handling('random_data_random_untrained_network', architecture, num=str(num), exps=exps, 
                                    pre_path=pre_path, 
                                    normal_dist=data_properties['normal_dist'], loc=data_properties['loc'], 
                                    scale=data_properties['scale'], bias=bias, 
                                    exp_type=data_properties['exp_type'], model_type=model_type,
                                    new_model_each_time=new_model_each_time)

    if os.path.isfile(f'{this_path}/all_gifs.gif'):
        return

    
    model = MLP_simple(n_in=architecture[0], layer_list=architecture[1], bias=bias)
    model.to(device)
    model.eval()

    feature_extractor = ReluExtractor(model, device=device)

    results_dict = {}
    
    if not new_data_each_time:
        x, y = create_random_data(input_dimension=architecture[0], num=num * exps, normal_dsit=data_properties['normal_dist'], loc=data_properties['loc'], scale=data_properties['scale'], exp_type=data_properties['exp_type'])

        fixed_model_batch_analysis(model, x, y, device, this_path, 'random', batch_size=num, plotting=True, no_labels=True)

        return

    for i in range(exps):
        if new_data_each_time:
            x, y = create_random_data(input_dimension=architecture[0], num=num, normal_dsit=data_properties['normal_dist'], loc=data_properties['loc'], scale=data_properties['scale'], exp_type=data_properties['exp_type'])
        if new_model_each_time:
            model.init_all_weights(device=device) # type: ignore

        model.to(device)
        relu_outputs = hook_forward_past(feature_extractor, x, y, device)

        results_dict = merge_results(all_analysis_for_hook_engine(relu_outputs, x), results_dict)

    plotting_actions(results_dict, num, this_path, architecture[1])

    plot_gifs(results_dict, this_path, custom_range=max(np.abs(data_properties['scale'] * 2), 10, int(np.abs(data_properties['loc'] / 2))), pre_path=this_path, scale=data_properties['scale'], eigenvectors=np.array(results_dict['eigenvectors'], dtype=object), num=num)
