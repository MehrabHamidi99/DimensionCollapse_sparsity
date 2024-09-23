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
  return stable_neuron_analysis(model, x, y, device, eval)

  
def one_random_experiment(architecture, exps=50, num=1000, one=True, return_sth=False, pre_path='', normal_dist=False, 
                          loc=0, scale=1, exp_type='normal', constant=5, projection_analysis_bool=False, stats=True, bias=1e-4, model_type='mlp', new_network_each_time=False):
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
                                 normal_dist=normal_dist, loc=loc, scale=scale, bias=bias, exp_type=exp_type, model_type=model_type)
      
  res_run1 = []
  # res_run2 = []
  eigens = []
  eigen_count = []
  # dist_all = np.zeros((len(architecture[1]) + 1, int(num * (num - 1) / 2)))
  dist_all = np.zeros((len(architecture[1]) + 1, num))

  dist_stats = []
  display_neuron_matrx_s = []
  stable_ranks_all = []
  simple_spherical_mean_width_all = []
  spherical_mean_width_v2_all = []
  cell_dims = []

  if model_type == 'mlp':
    net = MLP_ReLU(n_in=architecture[0], layer_list=architecture[1], bias=bias)
  else:
    net = ResNet_arch(n_in=architecture[0], layer_list=architecture[1], bias=bias)
  net.to(device)


  for i in tqdm(range(exps)):

    if new_network_each_time:
      if model_type == 'mlp':
        net = MLP_ReLU(n_in=architecture[0], layer_list=architecture[1], bias=bias)
      else:
        net = ResNet_arch(n_in=architecture[0], layer_list=architecture[1], bias=bias)
      net.to(device)
    
    r1, cell_dim, stable_ranks, simple_spherical_mean_width, spherical_mean_width_v2, count_num, eigen, dists, dist_stats_this = one_random_dataset_run(model=net, n=num, d=architecture[0], device=device,
                                    normal_dist=normal_dist, loc=loc, scale=scale,
                                    exp_type=exp_type, constant=constant, eval=False)
    res_run1 += [r1]
    eigens += [eigen]
    eigen_count += [count_num]
    dist_all = (dist_all + dists) / 2.0
    dist_stats += [dist_stats_this]
    stable_ranks_all += [stable_ranks]
    simple_spherical_mean_width_all += [simple_spherical_mean_width]
    spherical_mean_width_v2_all += [spherical_mean_width_v2]
    cell_dims += [cell_dim]
    # print(i)
    
    display_neuron_matrx_s += [net.analysis_neurons_activations_depth_wise()]
    net.reset()
    
  res1 = np.array(res_run1).mean(axis=0)
  eigen_count = np.array(eigen_count).mean(axis=0)
  eigens = np.array(eigens, dtype=object).mean(axis=0)
  dis_stats = np.array(dist_stats).mean(axis=0)
  display_neuron_matrx_s = np.array(display_neuron_matrx_s).mean(axis=0)
  stable_ranks_all = np.array(stable_ranks_all).mean(axis=0)
  simple_spherical_mean_width_all = np.array(simple_spherical_mean_width_all).mean(axis=0)
  spherical_mean_width_v2_all = np.array(spherical_mean_width_v2_all).mean(axis=0)
  cell_dims = np.array(cell_dims).mean(axis=0)


  plotting_actions(res1, stable_ranks_all, simple_spherical_mean_width_all, spherical_mean_width_v2_all, eigen_count, num, this_path, net, dis_stats, display_neuron_matrx_s, cell_dims)
  # plot_distances(net=net, distances=dis_stats, this_path=this_path)

  layer_activation_ratio = net.analysis_neurons_layer_wise_animation(res1, num)
  # if projection_analysis_bool:
  
  list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, eigen_vectors = one_random_dataset_run(model=net, n=num, d=architecture[0], device=device, normal_dist=normal_dist, loc=loc, scale=scale, exp_type=exp_type, constant=constant, eval=True)
    # projection_plots(list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, pre_path=this_path, costume_range=max(np.abs(scale * 2), 10, int(np.abs(loc / 2))))
    # animate_histogram(dist_all / max(1, np.mean(dist_all)), 'layers: ', x_axis_title='distance from origin distribution / mean', save_path='distance_distribution.gif', 
    #                   pre_path=this_path, fixed_scale=True, custom_range=scale, step=False)
  
    # animate_histogram(layer_activation_ratio, 'layer ', save_path='layer_wise_.gif', pre_path=this_path)
    # animate_histogram(eigens, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise.gif', pre_path=this_path, fixed_scale=False, custom_range=1, zero_one=False, eigens=True)  
  plot_gifs(layer_activation_ratio, eigens, dist_all, list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, this_path, costume_range=max(np.abs(scale * 2), 10, int(np.abs(loc / 2))), pre_path=this_path, scale=scale, eigenvectors=np.array(eigen_vectors, dtype=object))
  

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
  simple_model.extra()

  train_loader, val_loader, test_loader = get_mnist_data_loaders()

  train_x = train_loader.dataset.dataset.data[train_loader.dataset.indices,:,:].to(torch.float32).to(device).flatten(1)
  val_x = val_loader.dataset.dataset.data[val_loader.dataset.indices,:,:].to(torch.float32).to(device).flatten(1)
  test_x = test_loader.dataset.data.to(torch.float32).to(device).flatten(1)

  over_path = this_path + "untrained_"
  stable_mnist_analysis(simple_model, 'train', over_path=over_path, dataset_here=train_x, data_loader=train_loader, device=device)
  stable_mnist_analysis(simple_model, 'val', over_path=over_path, dataset_here=val_x, data_loader=val_loader, device=device)
  stable_mnist_analysis(simple_model, 'test', over_path=over_path, dataset_here=test_x, data_loader=test_loader, device=device)

  train_add, train_eig, val_add, val_eig = train_model(simple_model, train_loader, test_loader, base_path=this_path, train_x=train_x, val_x=val_x, val_loader=val_loader, epochs=epochs, loss='crossentropy')

  over_path = this_path + "trained_"
  stable_mnist_analysis(simple_model, 'train', over_path=over_path, dataset_here=train_x, data_loader=train_loader, device=device)
  stable_mnist_analysis(simple_model, 'val', over_path=over_path, dataset_here=val_x, data_loader=val_loader, device=device)
  stable_mnist_analysis(simple_model, 'test', over_path=over_path, dataset_here=test_x, data_loader=test_loader, device=device)

  # simple_model.to('cpu')
  animate_histogram(train_add, 'epoch ', save_path='epoch_visualization_train.gif', pre_path=this_path)
  animate_histogram(val_add, 'epoch ', save_path='epoch_visualization_val.gif', pre_path=this_path)

  animate_histogram(train_eig, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise_train.gif', pre_path=this_path, fixed_scale=True, custom_range=1)
  animate_histogram(val_eig, 'layers: ', x_axis_title='eigenvalues distribution', save_path='eigenvalues_layer_wise_val.gif', pre_path=this_path, fixed_scale=True, custom_range=1)


def random_experiment_hook_engine(architecture, exps=50, num=1000, pre_path='', data_properties={'normal_dist': True, 'loc': 0, 'scale': 1, 'exp_type': 'normal'}, bias=0, model_type='mlp', new_model_each_time=False):

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  this_path = file_name_handling('random_data_random_untrained_network', architecture, num=num, exps=exps, 
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
    train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = func_loader()
    
    over_path = this_path + "untrained_"
    fixed_model_batch_analysis_one_batch(model, train_samples, train_labels, device, '{}_{}'.format(over_path, 'train_'), '784, [256, 128, 64, 32, 10]')
    fixed_model_batch_analysis_one_batch(model, val_samples, val_labels, device, '{}_{}'.format(over_path, 'val_'), '784, [256, 128, 64, 32, 10]')
    fixed_model_batch_analysis_one_batch(model, test_samples, test_labels, device, '{}_{}'.format(over_path, 'test_'), '784, [256, 128, 64, 32, 10]')

    train_model(model, train_loader=train_loader, test_loader=test_loader, base_path=this_path, train_x=train_samples, train_y=train_labels, val_x=val_samples, val_y=val_labels, val_loader=val_loader, test_x=test_samples, test_y=test_labels, epochs=epochs, loss='crossentropy')    
    
    create_gif_from_plots(this_path, f'{this_path}/train_plots_animation.gif', plot_type='train')
    create_gif_from_plots(this_path, f'{this_path}/val_plots_animation.gif', plot_type='val')



def cifar_data():

  # Step 1: Define device (GPU/CPU)
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Step 2: Define the path for saving results
  pre_path = 'cifar_results'
  try_num = 1  # Example
  this_path = f'{pre_path}/pretrained_model_analysis/try_num{try_num}/'
  if not os.path.isdir(this_path):
      os.makedirs(this_path)

  # Step 3: Load Pre-trained Model (e.g., ResNet18)
  model = models.resnet18(pretrained=True)  # You can replace with other models
  model.to(device)  # Move model to device (GPU/CPU)

  # Step 4: Modify model for binary classification
  # ResNet18 has a fully connected layer (fc) that outputs 1000 classes by default
  # We change it to output 2 classes (for binary classification)
  num_classes = 2
  model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
  model.to(device)

  
  # Directory to save the samples
  save_dir = './saved_data'
  os.makedirs(save_dir, exist_ok=True)

  train_samples_path = os.path.join(save_dir, 'train_samples.pt')
  train_labels_path = os.path.join(save_dir, 'train_labels.pt')
  test_samples_path = os.path.join(save_dir, 'test_samples.pt')
  test_labels_path = os.path.join(save_dir, 'test_labels.pt')

  # Check if samples are already saved, load them
  if os.path.exists(train_samples_path) and os.path.exists(train_labels_path) and \
    os.path.exists(test_samples_path) and os.path.exists(test_labels_path):

      print("Loading saved train and test samples...")

      # Load saved data
      train_samples = torch.load(train_samples_path)
      train_labels = torch.load(train_labels_path)
      test_samples = torch.load(test_samples_path)
      test_labels = torch.load(test_labels_path)

  else:
      print("Processing CIFAR-10 dataset...")

      # Step 1: Load CIFAR-10 dataset
      transform = transforms.Compose([
          transforms.Resize((224, 224)),  # Resize for models like ResNet
          transforms.ToTensor(),
          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
      ])

      full_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

      # Step 2: Filter dataset for two classes (cats = 3, dogs = 5) without looping
      class_map = {3: 0, 5: 1}  # Map cat to 0 and dog to 1
      target_np = np.array(full_dataset.targets)  # Convert targets to numpy array

      # Get indices for cats and dogs
      filtered_indices = np.where((target_np == 3) | (target_np == 5))[0]

      # Get filtered targets using NumPy's vectorized operations
      filtered_targets = np.vectorize(class_map.get)(target_np[filtered_indices])

      # Step 3: Create a Subset dataset using the filtered indices
      filtered_dataset = Subset(full_dataset, filtered_indices)

      # Assign filtered targets to the dataset (binary labels)
      filtered_dataset.targets = torch.tensor(filtered_targets)

      # Step 4: Split into training and testing sets
      train_size = int(0.8 * len(filtered_dataset))
      test_size = len(filtered_dataset) - train_size
      train_dataset, test_dataset = torch.utils.data.random_split(filtered_dataset, [train_size, test_size])

      # Step 5: Create data loaders
      train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
      test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

      # Step 6: Extract samples and labels without loops
      train_samples, train_labels = next(iter(train_loader))
      test_samples, test_labels = next(iter(test_loader))

      # Move the samples and labels to the appropriate device (GPU/CPU)
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      train_samples = train_samples.to(device)
      train_labels = train_labels.to(device)
      test_samples = test_samples.to(device)
      test_labels = test_labels.to(device)

      # Step 7: Save the processed data for future use
      print("Saving processed train and test samples...")
      torch.save(train_samples, train_samples_path)
      torch.save(train_labels, train_labels_path)
      torch.save(test_samples, test_samples_path)
      torch.save(test_labels, test_labels_path)

  print(f"Train samples shape: {train_samples.shape}")
  print(f"Train labels shape: {train_labels.shape}")
  print(f"Test samples shape: {test_samples.shape}")
  print(f"Test labels shape: {test_labels.shape}")


  # Step 8: Run analysis on the pre-trained model for binary classification
  over_path = this_path + "pretrained_binary_"
  fixed_model_batch_analysis(
      model, train_samples, train_labels, device, 
      '{}_{}'.format(over_path, 'train_'), '224, [512, 256, 128, 64, 2]'
  )
  fixed_model_batch_analysis(
      model, test_samples, test_labels, device, 
      '{}_{}'.format(over_path, 'test_'), '224, [512, 256, 128, 64, 2]'
  )

