from utils import *
import torch.nn as nn
import numpy as np

from Models import *
from DataGenerator import *
from utils import *
from Training import *


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

def one_random_dataset_run(model, n, d, normal_dist=False, loc=0, scale=1):
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
  x, y = create_random_data(d, n, normal_dsit=normal_dist, loc=loc, scale=scale)
  stable_neuron_analysis(model, x)

  return model.additive_activations, model.additive_activation_ratio


def one_random_experiment(architecture, exps=500, num=1000, one=True, return_sth=False, pre_path='', normal_dist=False, loc=0, scale=1):
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
  if normal_dist:
    pre_path += 'normal_std{}/'.format(str(scale))
  else:
    pre_path += 'uniform/'
  this_path = pre_path + 'random_data_random_untrained_network{}_exps{}_num{}/'.format(str(architecture), str(exps), str(num))
  if not os.path.isdir(this_path):
    try:
      os.makedirs(this_path)
    except OSError as exc:
      this_path = pre_path + 'random_data_random_untrained_network{}_exps{}_num{}/'.format(str(len(architecture[1])), str(exps), str(num))
      if not os.path.isdir(this_path):
        os.makedirs(this_path)
      
  res_run1 = []
  res_run2 = []
  net = MLP_ReLU(n_in=architecture[0], layer_list=architecture[1])
  for i in range(exps):
    r1, r2 = one_random_dataset_run(net, 1000, architecture[0], normal_dist, loc, scale)
    res_run1 += [r1]
    res_run2 += [r2]
    net.reset()

  res1 = np.array(res_run1).mean(axis=0)
  # res2 = np.array(res_run2).mean(axis=0)

  fig, ax = plt.subplots(figsize=(10, 10))
  tp = ax.hist(res1 / num, bins=20)
  ax.set_xlabel('neuron activation percentage of a given dataset')
  ax.set_ylabel('neuron frequency')
  ax.set_title('#neurons:{}, #layers:{}'.format(str(np.sum(net.layer_list)), str(len(net.layer_list))))
  fig.savefig(this_path + 'additive_activations.pdf')
  plt.close(fig)

  layer_activation_ratio = net.analysis_neurons_layer_wise_animation(res1, num)
  animate_histogram(layer_activation_ratio, 'layer ', save_path='layer_wise.gif', pre_path=pre_path)

  # fig, ax = plt.subplots(figsize=(10, 10))
  # tp = ax.bar(np.arange(len(net.get_layer_list())), res2)
  # ax.set_xticks(np.arange(len(net.get_layer_list()) + 1))
  # ax.set_xticklabels([''] + ['layer' + str(i + 1) for i in np.arange(len(net.get_layer_list()))])
  # fig.savefig(this_path + 'additive_activation_ratio.pdf')
  if return_sth:
    return res_run1, res_run2, net
  
  return res_run1

def before_after_training_experiment(architecture):
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
  
  this_path = 'random_data_random_trained_network{}/'.format(str(architecture))
  if not os.path.isdir(this_path):
    os.makedirs(this_path)
  n_in = architecture[0]
  simple_model = MLP_ReLU(n_in, layer_list=architecture[1])
  train, val, test = create_full_random_data(n_in)
  train_loader = get_data_loader(train[0], train[1])
  val_loader = get_data_loader(val[0], val[1], batch_size=1)

  stable_neuron_analysis(simple_model, train[0])

  fig, ax = plt.subplots(figsize=(10, 10))
  tp = ax.hist(simple_model.additive_activations / train[0].shape[0], bins=10)
  fig.savefig(this_path + 'before_trainig_additive_activations.pdf')
  plt.close()
  plt.cla()
  plt.clf()

  fig, ax = plt.subplots(figsize=(10, 10))
  tp = ax.bar(np.arange(len(simple_model.get_layer_list())), simple_model.additive_activation_ratio / train[0].shape[0])
  fig.savefig(this_path + 'before_trainig_additive_activations_ratio.pdf')
  plt.close()
  plt.cla()
  plt.clf()

  v, va = train_model(simple_model, train_loader, val_loader)
  stable_neuron_analysis(simple_model, train[0])


  fig, ax = plt.subplots(figsize=(10, 10))
  tp = ax.hist(simple_model.additive_activations / train[0].shape[0], bins=10)
  fig.savefig(this_path + 'after_trainig_additive_activations.pdf')
  plt.close()
  plt.cla()
  plt.clf()
  fig, ax = plt.subplots(figsize=(10, 10))
  tp = ax.bar(np.arange(len(simple_model.get_layer_list())), simple_model.additive_activation_ratio / train[0].shape[0])
  fig.savefig(this_path + 'after_trainig_additive_activations_ratio.pdf')
  plt.close()
  plt.cla()
  plt.clf()
