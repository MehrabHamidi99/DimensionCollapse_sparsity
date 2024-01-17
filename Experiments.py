from Models import *
from DataGenerator import *
from utils import *
from Training import *
from ForwardPass import *

def one_random_dataset_run(model, n, d, device, normal_dist=False, loc=0, scale=1, exp_type='normal', eval=False, constant=5):
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
  x, y = create_random_data(input_dimension=d, num=n, normal_dsit=normal_dist, loc=loc, scale=scale, exp_type=exp_type, constant=constant)
  # stable_neuron_analysis(model, x, device)
  return stable_neuron_analysis(model, x, y, device, eval)

  
def one_random_experiment(architecture, exps=50, num=1000, one=True, return_sth=False, pre_path='', normal_dist=False, 
                          loc=0, scale=1, exp_type='normal', constant=5, projection_analysis_bool=False, stats=True, bias=1e-4):
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
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  this_path = file_name_handling('random_data_random_untrained_network', architecture, num=num, exps=exps, pre_path=pre_path, 
                                 normal_dist=normal_dist, loc=loc, scale=scale, bias=bias)
      
  res_run1 = []
  # res_run2 = []
  # eigens = []
  eigen_count = []
  # dist_all = np.zeros((len(architecture[1]) + 1, int(num * (num - 1) / 2)))
  dist_all = np.zeros((len(architecture[1]) + 1, num))

  dist_stats = []

  net = MLP_ReLU(n_in=architecture[0], layer_list=architecture[1], bias=bias)
  net.to(device)
  for i in range(exps):
    r1, count_num, dists, dist_stats_this = one_random_dataset_run(model=net, n=num, d=architecture[0], device=device,
                                    normal_dist=normal_dist, loc=loc, scale=scale,
                                    exp_type=exp_type, constant=constant, eval=False)
    
    res_run1 += [r1]
    # eigens += [eigens]
    eigen_count += [count_num]
    dist_all = (dist_all + dists) / 2.0
    dist_stats += [dist_stats_this]
    net.reset()
  
  _, _, eigens, list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, distances, dis_stats = one_random_dataset_run(model=net, n=num, d=architecture[0], device=device,
                                    normal_dist=normal_dist, loc=loc, scale=scale,
                                    exp_type=exp_type, constant=constant, eval=True)

  res1 = np.array(res_run1).mean(axis=0)
  eigen_count = np.array(eigen_count).mean(axis=0)
  dis_stats = np.array(dist_stats).mean(axis=0)

  plotting_actions(res1, eigen_count, num, this_path, net)
  layer_activation_ratio = net.analysis_neurons_layer_wise_animation(res1, num)
  animate_histogram(layer_activation_ratio, 'layer ', save_path='layer_wise_.gif', pre_path=this_path)
  animate_histogram(eigens, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise.gif', pre_path=this_path, fixed_scale=True, custom_range=1)
  if projection_analysis_bool:
    projection_plots(list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, pre_path=this_path)
    animate_histogram(distances, 'layers: ', x_axis_title='distance from origin distribution / mean', save_path='distance_distribution.gif', 
                      pre_path=this_path, fixed_scale=True, custom_range=scale * 2.5, step=False)
    if stats:
      plot_distances(net=net, distances=dis_stats, this_path=this_path)

  if return_sth:
    return res1, net
  return res1

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



def mnist_training_analysis(architecture, epochs=50, pre_path=''):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  this_path = file_name_handling('mnist_analysis', architecture, pre_path=pre_path)

  simple_model = MLP_mnist(hiddens=architecture).to(device)

  train_loader, val_loader, test_loader = get_mnist_data_loaders()

  train_x = train_loader.dataset.dataset.data[train_loader.dataset.indices,:,:].to(torch.float32).to(device).flatten(1)
  val_x = val_loader.dataset.dataset.data[val_loader.dataset.indices,:,:].to(torch.float32).to(device).flatten(1)
  test_x = test_loader.dataset.data.to(torch.float32).to(device).flatten(1)

  over_path = this_path + "untrained_"
  whole_data_analysis_forward_pass(simple_model, 'train', over_path=over_path, dataset_here=train_x)
  whole_data_analysis_forward_pass(simple_model, 'val', over_path=over_path, dataset_here=val_x)
  whole_data_analysis_forward_pass(simple_model, 'test', over_path=over_path, dataset_here=test_x)

  train_add, train_eig, val_add, val_eig = train_model(simple_model, train_loader, test_loader, base_path=this_path, train_x=train_x, val_x=val_x, val_loader=val_loader, epochs=epochs, loss='crossentropy')

  over_path = this_path + "trained_"
  whole_data_analysis_forward_pass(simple_model, 'train', over_path=over_path, dataset_here=train_x)
  whole_data_analysis_forward_pass(simple_model, 'val', over_path=over_path, dataset_here=val_x)
  whole_data_analysis_forward_pass(simple_model, 'test', over_path=over_path, dataset_here=test_x)

  # simple_model.to('cpu')
  animate_histogram(train_add, 'epoch ', save_path='epoch_visualization_train.gif', pre_path=this_path)
  animate_histogram(val_add, 'epoch ', save_path='epoch_visualization_val.gif', pre_path=this_path)
  # animate_histogram(train_eig, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise_train.gif', pre_path=this_path, fixed_scale=True, custom_range=1)
  # animate_histogram(val_eig, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise_val.gif', pre_path=this_path, fixed_scale=True, custom_range=1)