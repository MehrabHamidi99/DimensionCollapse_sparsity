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



def cifar100_training_analysis_hook_engine(try_num, archirecture=(3*32*32, [256, 128, 64, 32, 100]), epochs=100, pre_path='', bias=1e-4, debug=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/cifar100_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)


    func_loader = get_cifar100_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    if not debug:
        wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

        # Initialize a wandb run
        wandb.init(project="{}_{}".format(pre_path.replace("\\", "_").replace("/", "_"), str(try_num)))
        # Enable logging of GPU utilization
        wandb.config.update({"track_gpu": True})

    # over_path = this_path + "untrained_"
    # fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), 'arch')
    # fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), 'arch')
    # fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), 'arch')

    train_model(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')