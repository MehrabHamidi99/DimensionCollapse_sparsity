from Models import *
from DataGenerator import *
from utils import *
from Training import *
from ForwardPass import *
from Models_normal import *
from FeatureExtractor import *
from Analysis import *
import torch
import os
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
from VisionTranformer import VisionTransformer

import wandb
from other_models import MLPMixer

def random_experiment_hook_engine(architecture, exps=50, num=1000, pre_path='', data_properties={'normal_dist': True, 'loc': 0, 'scale': 1, 'exp_type': 'normal'}, bias=0, model_type='mlp', new_model_each_time=False):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  this_path = file_name_handling('random_data_random_untrained_network', architecture, num=str(num), exps=exps, 
                                 pre_path=pre_path, 
                                 normal_dist=data_properties['normal_dist'], loc=data_properties['loc'], 
                                 scale=data_properties['scale'], bias=bias, 
                                 exp_type=data_properties['exp_type'], model_type=model_type,
                                 new_model_each_time=new_model_each_time)
  
  model = MLP_simple(n_in=architecture[0], layer_list=architecture[1], bias=bias)

  feature_extractor = ReluExtractor(model, device=device)

  results_dict = {}

  for i in range(exps):
      if new_model_each_time:
          model.init_all_weights()
      x, y = create_random_data(input_dimension=architecture[0], num=num, normal_dsit=data_properties['normal_dist'], loc=data_properties['loc'], scale=data_properties['scale'], exp_type=data_properties['exp_type'])

      relu_outputs = hook_forward(feature_extractor, x, y, device)

      results_dict = merge_results(all_analysis_for_hook_engine(relu_outputs, x), results_dict)

  plotting_actions(results_dict, num, this_path, architecture[1])

  plot_gifs(results_dict, this_path, costume_range=max(np.abs(data_properties['scale'] * 2), 10, int(np.abs(data_properties['loc'] / 2))), pre_path=this_path, scale=data_properties['scale'], eigenvectors=np.array(results_dict['eigenvectors'], dtype=object), num=num)
  

def batch_fixed_model_hook_engine(architecture, data_loader, data_properties, num):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  x, y = create_random_data(input_dimension=architecture[0], num=num, normal_dsit=data_properties['normal_dist'], loc=data_properties['loc'], scale=data_properties['scale'], exp_type=data_properties['exp_type'])

  model = MLP_simple(n_in=architecture[0], layer_list=architecture[1], bias=0)
  fixed_model_batch_analysis(model, x, y, device, 'save_path', 'model_status')



def mnist_training_analysis_spike_loss(try_num, archirecture=(784, [256, 128, 64, 64, 64, 64, 32, 10]), epochs=100, pre_path='', bias=1e-4, three_class=False, odd_even=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/mnist_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)

    if odd_even:
      func_loader = get_mnist_data_loaders_odd_even
    elif three_class:
      func_loader = get_mnist_data_loaders_three_class
    else:
      func_loader = get_mnist_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=32)
    
    wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

    # Initialize a wandb run
    wandb.init(project="{}_{}".format(pre_path, str(try_num)))
    # Enable logging of GPU utilization
    wandb.config.update({"track_gpu": True})

    # over_path = this_path + "untrained_"
    # fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), '784, [256, 128, 64, 32, 10]')
    # fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), '784, [256, 128, 64, 32, 10]')
    # fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    train__with_spike_loss(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')


def mnist_training_analysis_hook_engine(try_num, archirecture=(784, [256, 128, 64, 32, 10]), epochs=100, pre_path='', bias=1e-4, three_class=False, odd_even=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/mnist_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)

    if odd_even:
      func_loader = get_mnist_data_loaders_odd_even
    elif three_class:
      func_loader = get_mnist_data_loaders_three_class
    else:
      func_loader = get_mnist_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=32)
    
    wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

    # Initialize a wandb run
    wandb.init(project="{}_{}".format(pre_path, str(try_num)))
    # Enable logging of GPU utilization
    wandb.config.update({"track_gpu": True})

    over_path = this_path + "untrained_"
    fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), '784, [256, 128, 64, 32, 10]')
    fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), '784, [256, 128, 64, 32, 10]')
    fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    train_model(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')



def fashion_mnist_training_analysis_hook_engine(try_num, archirecture=(784, [256, 128, 64, 32, 10]), epochs=100, pre_path='', bias=1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/fashion_mnist_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)


    func_loader = get_fashion_mnist_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

    # Initialize a wandb run
    wandb.init(project="{}_{}".format(pre_path, str(try_num)))
    # Enable logging of GPU utilization
    wandb.config.update({"track_gpu": True})

    over_path = this_path + "untrained_"
    fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), '784, [256, 128, 64, 32, 10]')
    fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), '784, [256, 128, 64, 32, 10]')
    fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    train_model(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')


def cifar10_training_analysis_spike_loss(try_num, archirecture=(3*32*32, [256, 128, 64, 32, 10]), epochs=100, pre_path='', bias=1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/cifar_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)


    func_loader = get_cifar10_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

    # Initialize a wandb run
    wandb.init(project="{}_{}".format(pre_path, str(try_num)))
    # Enable logging of GPU utilization
    wandb.config.update({"track_gpu": True})

    # over_path = this_path + "untrained_"
    # fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), '784, [256, 128, 64, 32, 10]')
    # fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), '784, [256, 128, 64, 32, 10]')
    # fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    train__with_spike_loss(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')

def cifar10_training_analysis_hook_engine(try_num, archirecture=(3*32*32, [256, 128, 64, 32, 10]), epochs=100, pre_path='', bias=1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/cifar_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)


    func_loader = get_cifar10_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

    # Initialize a wandb run
    wandb.init(project="{}_{}".format(pre_path, str(try_num)))
    # Enable logging of GPU utilization
    wandb.config.update({"track_gpu": True})

    over_path = this_path + "untrained_"
    fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), '784, [256, 128, 64, 32, 10]')
    fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), '784, [256, 128, 64, 32, 10]')
    fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    train_model(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')



def cifar100_training_analysis_hook_engine(try_num, archirecture=(3*32*32, [256, 128, 64, 32, 100]), epochs=100, pre_path='', bias=1e-4, debug=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/cifar_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)


    func_loader = get_cifar100_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    if not debug:
        wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

        # Initialize a wandb run
        wandb.init(project="{}_{}".format(pre_path, str(try_num)))
        # Enable logging of GPU utilization
        wandb.config.update({"track_gpu": True})

    # over_path = this_path + "untrained_"
    # fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), 'arch')
    # fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), 'arch')
    # fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), 'arch')

    train_model(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')


def cifar10_training_analysis_hook_engine_resnetMLP(try_num, epochs=100, pre_path='', bias=1e-4, debug=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/cifar_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = 4,
        dim = 512,
        depth = 12,
        num_classes = 10
    ).to(device)


    func_loader = get_cifar10_data_loaders_resnetMLP
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    if not debug:
        wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

        # Initialize a wandb run
        wandb.init(project="{}_{}".format(pre_path, str(try_num)))
        # Enable logging of GPU utilization
        wandb.config.update({"track_gpu": True})

    analyze_b_zie = 1000

    over_path = this_path + "untrained_"
    fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), 'res_net_mlp', batch_size=analyze_b_zie)
    fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), 'res_net_mlp', batch_size=analyze_b_zie)
    fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), 'res_net_mlp', batch_size=analyze_b_zie)

    train_model(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy', optimizer='adamw', analye_b_size=analyze_b_zie)    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')


def viT_mnist(try_num, epochs=100, pre_path='', bias=1e-4, debug=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/cifar_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = VisionTransformer(n_channels=1,   embed_dim=64, 
                                       n_layers=6, n_attention_heads=4, 
                                       forward_mul=2, image_size=28, 
                                       patch_size=4, n_classes=10, 
                                       dropout=0.1)


    func_loader = get_mnist_data_loaders_cnn_based_model
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    if not debug:
        wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

        # Initialize a wandb run
        wandb.init(project="{}_{}".format(pre_path, str(try_num)))
        # Enable logging of GPU utilization
        wandb.config.update({"track_gpu": True})

    analyze_b_zie = 10000

    over_path = this_path + "untrained_"
    fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), 'res_net_mlp', batch_size=analyze_b_zie)
    fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), 'res_net_mlp', batch_size=analyze_b_zie)
    fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), 'res_net_mlp', batch_size=analyze_b_zie)

    train_model(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy', optimizer='adamw', analye_b_size=analyze_b_zie)    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')



def kuji_mnist_training_analysis_spike_loss(try_num, archirecture=(784, [256, 128, 64, 64, 64, 64, 32, 10]), epochs=100, pre_path='', bias=1e-4, three_class=False, odd_even=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/mnist_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)

    func_loader = get_kuzushiji_mnist_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

    # Initialize a wandb run
    wandb.init(project="{}_{}".format(pre_path, str(try_num)))
    # Enable logging of GPU utilization
    wandb.config.update({"track_gpu": True})

    # over_path = this_path + "untrained_"
    # fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), '784, [256, 128, 64, 32, 10]')
    # fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), '784, [256, 128, 64, 32, 10]')
    # fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    train__with_spike_loss(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')



def three_class_cifar10_training_analysis_spike_loss(try_num, archirecture=(3*32*32, [256, 128, 64, 32, 10]), epochs=100, pre_path='', bias=1e-4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    this_path = '{}/cifar_training/try_num{}/'.format(pre_path, str(try_num))
    if not os.path.isdir(this_path):
        os.makedirs(this_path)

    model = MNIST_classifier(n_in=archirecture[0], layer_list=archirecture[1], bias=bias)


    func_loader = get_three_class_cifar10_data_loaders
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader(batch_size=64)
    
    wandb.login(key="6e39572f3cebe5c6b8020ae79454587397fd5f43")

    # Initialize a wandb run
    wandb.init(project="{}_{}".format(pre_path, str(try_num)))
    # Enable logging of GPU utilization
    wandb.config.update({"track_gpu": True})

    # over_path = this_path + "untrained_"
    # fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), '784, [256, 128, 64, 32, 10]')
    # fixed_model_batch_analysis(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), '784, [256, 128, 64, 32, 10]')
    # fixed_model_batch_analysis(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    train__with_spike_loss(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')
