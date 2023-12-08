from utils import *
import torch.nn as nn
import numpy as np

from Models import *
from DataGenerator import *
from utils import *
from Training import *


##########
def stable_neuron_analysis(model, dataset, y=None):
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
    # Process each input in the dataset
    for x in dataset:
        # Get pre-activation and activation values for each layer
        _ = model(torch.tensor(x, dtype=torch.float32), return_pre_activations=True)
        model.analysis_neurons_activations_depth_wise(dataset.shape[0])

def one_random_dataset_run(model, n, d, normal_dist=False, loc=0, scale=1, return_eigenvalues=False, necc=False, this_path='', exp_type='normal'):
  '''
    Parameters
    model (torch.nn.Module): A trained PyTorch neural network model.
    n (int): The number of data points to generate for the dataset.
    d (int): The number of features in each data point.
    
    Returns
    A tuple of (additive_activations, additive_activation_ratio) representing the aggregated neuron activations and their ratios.
    
    Functionality
    Generates a random dataset using create_random_data.
    Analyzes the stable neuron activation using stable_neuron_analysis.
    Returns the aggregated activations and activation ratios from the model.

  '''
  # visualize_1D_boundaries(net)
  x, y = create_random_data(d, n, normal_dsit=normal_dist, loc=loc, scale=scale, exp_type=exp_type)
  stable_neuron_analysis(model, x)
  if not necc:
    return model.additive_activations, model.additive_activation_ratio
  res = model.other_forward_analysis(x, return_eigenvalues)

  if return_eigenvalues:
    return model.additive_activations, model.additive_activation_ratio, res[0], res[1], res[2], res[3], res[4], res[5], res[6]
  return model.additive_activations, model.additive_activation_ratio, res, res[1], res[2], res[3], res[4], res[5]

def one_random_experiment(architecture, exps=500, num=1000, one=True, return_sth=False, pre_path='', normal_dist=False, loc=0, scale=1, exp_type='normal'):
  '''
    Parameters
    architecture (tuple): A tuple where the first element is the number of input features, and the second element is a list of layer sizes for the network.
    exps (int, optional): The number of experiments to run. Default is 500.
    num (int, optional): The number of data points in each dataset. Default is 1000.
    one (bool, optional): Flag to control certain processing, not explicitly defined in the function.
    return_sth (bool, optional): If True, returns the results of the experiments.
    
    Functionality
    Iterates over the specified number of experiments.
    In each experiment, it generates a random dataset and analyzes neuron activations.
    Aggregates and visualizes the results of these experiments.
    Optionally saves the aggregated results to a specified path.
    
    Usage
    Used for conducting large-scale experiments to understand the behavior of different network architectures on random data.
  '''

  this_path = file_name_handling('random_data_random_untrained_network', architecture, exps, num, pre_path=pre_path, normal_dist=normal_dist, loc=loc, scale=scale)
      
  res_run1 = []
  res_run2 = []
  # eigens = []
  eigen_count = []
  net = MLP_ReLU(n_in=architecture[0], layer_list=architecture[1])
  
  for i in range(exps):
    # r1, r2, count_num = one_random_dataset_run(net, num, architecture[0], normal_dist, loc, scale)
    r1, r2 = one_random_dataset_run(net, num, architecture[0], normal_dist, loc, scale, exp_type)
    res_run1 += [r1]
    res_run2 += [r2]
    # eigens += [eigens]
    # eigen_count += [count_num]
    net.reset()
  
  _, _, eigen_count, eigens, list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, distances = one_random_dataset_run(
                                    architecture[0], normal_dist, loc, scale, 
                                    return_eigenvalues=True, necc=True, this_path=this_path, exp_type=exp_type)
  res1 = np.array(res_run1).mean(axis=0)
  # eigen_count = np.array(eigen_count).mean(axis=0)

  plotting_actions(res1, eigen_count, num, this_path, net)

  layer_activation_ratio = net.analysis_neurons_layer_wise_animation(res1, num)
  animate_histogram(layer_activation_ratio, 'layer ', save_path='layer_wise_.gif', pre_path=this_path)
  animate_histogram(eigens, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise.gif', pre_path=this_path)
  projection_plots(list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, pre_path=this_path)
  animate_histogram(distances, 'layers: ', x_axis_title='pairwise distances distribution', save_path='distance_distribution.gif', pre_path=this_path, fixed_scale=True)

  if return_sth:
    return res_run1, res_run2, net
  return res1 / num

def before_after_training_experiment(architecture, num=1000, epochs=50, pre_path='', normal_dist=False, loc=0, scale=1):
  '''
    Parameters
    architecture (tuple): A tuple where the first element is the number of input features, 
    and the second element is a list of layer sizes for the neural network.
    
    Functionality
    Directory Setup: Creates a directory for saving the output visualizations based on the provided network architecture.
    Model Initialization: Initializes an MLP_ReLU model using the given architecture.
    Data Preparation: Generates training, validation, and test datasets 
    using create_full_random_data and prepares DataLoaders for them.
    Pre-Training Analysis: Performs neuron activation analysis on the training data using the untrained
      model and saves the visualizations of neuron activations.
    Model Training: Trains the model using the training and validation datasets.
    Post-Training Analysis: Performs the same neuron activation analysis on the training data using the trained model
      and saves the visualizations.
    Visualization: Histograms and bar plots are generated and saved to illustrate the distribution of 
    neuron activations and their ratios before and after training.
    
    Usage Example
    architecture = (10, [20, 30, 40])
    before_after_training_experiment(architecture)
    Visualizations
    Histograms and bar plots are saved to show the distribution of neuron 
    activations and their ratios before and after training. This is useful for comparing the effects of training on neuron behavior.
    
    Notes
    This function is particularly useful for research and studies 
    focused on understanding the impact of training on neural network internals.
    The function assumes the presence of certain methods in the MLP_ReLU model, 
    including get_layer_list and analysis_neurons_activations_depth_wise.
    Ensure that matplotlib and other required libraries are installed and properly configured, 
    especially if running in environments like Jupyter notebooks.
    This function provides a comprehensive means of analyzing and visualizing the change 
    in neuron activation patterns in a neural network due to training, 
    offering valuable insights into the model's learning process.
  '''
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  this_path = file_name_handling('random_data_random_trained_network', architecture, num, pre_path=pre_path, normal_dist=normal_dist, loc=loc, scale=scale)

  n_in = architecture[0]
  simple_model = MLP_ReLU(n_in, layer_list=architecture[1])
  train, val, test = create_full_random_data(n_in, output_dim=architecture[1][-1], train_num=int(num * 0.7), val_num=int(num * 0.2), test_num=num - (int(num * 0.7) + int(num * 0.3)), normal_dsit=normal_dist, loc=loc, scale=scale)
  
  train_loader = get_data_loader(train[0], train[1], batch_size=32)
  val_loader = get_data_loader(val[0], val[1], batch_size=1)

  stable_neuron_analysis(simple_model, train[0])
  eigen_count, eigens = simple_model.other_forward_analysis(train[0], True)

  plotting_actions(simple_model.additive_activations, eigen_count, train[0].shape[0], this_path, simple_model, suffix='training_data_before_training_')

  layer_activation_ratio = simple_model.analysis_neurons_layer_wise_animation(simple_model.additive_activations, train[0].shape[0])
  animate_histogram(layer_activation_ratio, 'layer ', save_path='layer_wise_pretraining_training_data.gif', pre_path=this_path)
  animate_histogram(eigens, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise.gif', pre_path=this_path)

  simple_model.reset_all()

  va = train_model(simple_model, train_loader, val_loader, epochs=epochs, validation_data=torch.tensor(val[0]))
  simple_model.to('cpu')
  animate_histogram(va, 'epoch ', save_path='epoch_visualization.gif', pre_path=this_path)
  stable_neuron_analysis(simple_model, train[0])
  eigen_count, eigens = simple_model.other_forward_analysis(train[0], True)

  plotting_actions(simple_model.additive_activations, eigen_count, train[0].shape[0], this_path, simple_model, suffix='training_data_after_training_')

  layer_activation_ratio = simple_model.analysis_neurons_layer_wise_animation(simple_model.additive_activations, train[0].shape[0])
  animate_histogram(layer_activation_ratio, 'layer ', save_path='layer_wise_after_training.gif', pre_path=this_path)
  animate_histogram(eigens, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise.gif', pre_path=this_path)

  return simple_model.additive_activations / train[0].shape[0]

