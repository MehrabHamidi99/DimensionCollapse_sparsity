import torch
import numpy as np
from DataLoader import *
from utils import *


def stable_neuron_analysis(model, dataset, labels, device, eval):
    '''
    Parameters
    model (torch.nn.Module): A trained PyTorch neural network model.
    dataset (iterable): A dataset where each element is an input sample to the model.
    y (iterable, optional): Output labels for the dataset, not used in the current implementation.
    
    Functionality
    Iterates over each input sample in the dataset.
    Performs a forward pass through the model to obtain pre-activation and activation values for each layer.
    Invokes analysis_neurons_activations_depth_wise on the model to analyze neuron activations.
    
    Usage
    Primarily used for understanding which neurons are consistently active or inactive across a dataset.
    '''
    
    model.to(device)
    if eval:
        model.extra()
    # Process each input in the dataset
    data_loader = get_data_loader(dataset, labels, batch_size=dataset.shape[0])
    for x, _ in data_loader:
        x_ = x.to(device)
        # Get pre-activation and activation values for each layer
        _ = model(x_)
    # return model.analysis_neurons_activations_depth_wise(dataset.shape[0])
    return model.post_forward_neuron_activation_analysis(x)


def whole_data_analysis_forward_pass(model, mode, over_path, dataset_here, scale=1):
    model.reset()
    model.extra()
    model(dataset_here)
    # model.extra()
    additive_act, eigenvalues_count, eigens, list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, distances, dis_stats = model.post_forward_neuron_activation_analysis(dataset_here)
    this_path = over_path + mode + '_'
    def do_all(additive_act, eigenvalues_count, eigens, list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, distances, dis_stats):
        plotting_actions(additive_act, eigenvalues_count, dataset_here.shape[0], this_path, model)
        layer_activation_ratio = model.analysis_neurons_layer_wise_animation(additive_act, dataset_here.shape[0])
        animate_histogram(layer_activation_ratio, 'layer ', save_path='layer_wise_.gif', pre_path=this_path)
        animate_histogram(eigens, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise.gif', pre_path=this_path, fixed_scale=True, custom_range=1)
        projection_plots(list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, pre_path=this_path)
        animate_histogram(distances, 'layers: ', x_axis_title='pairwise distances distribution / mean', save_path='distance_distribution.gif', pre_path=this_path, fixed_scale=True, custom_range=scale * 2.5, step=False)
        plot_distances(net=model, distances=dis_stats, this_path=this_path)

    do_all(additive_act, eigenvalues_count, eigens, list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, distances, dis_stats)
    return additive_act, eigenvalues_count, eigens
    
