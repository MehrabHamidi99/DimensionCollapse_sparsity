from utils import *
from abc import ABC, abstractmethod
from utils import nn
from utils import np
from sklearn.manifold import TSNE
import random
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist

class ParentNetwork(nn.Module, ABC):
    '''
    ParentNetwork is a custom neural network class that extends torch.nn.Module. 

    Parameters:
    n_in (int): Number of input features.
    layer_list (list of int): List containing the size of each layer in the network.

    Attributes:
    input_dim (int): Stores the size of the input layer.
    layer_list (np.array): An array representing the structure of the network including the input layer.
    additive_activation_ratio (np.array): Array to track the ratio of active neurons in each layer.
    additive_in_activation_ratio (np.array): Array to track the ratio of inactive neurons in each layer.
    first_layer (nn.Sequential): The first linear layer of the network.
    hidden_layers (nn.Sequential): Sequential container of hidden layers.
    output_dim (int): Size of the output layer.
    neurons_activations (np.array): Array to store activation status of neurons.
    additive_activations (np.array): Array to accumulate activations over inputs or epochs.


    Usage Example

    model = MLP_ReLU(n_in=10, layer_list=[20, 30, 40])
    output = model(torch.randn(10))

    '''
    def __init__(self, n_in, layer_list):
      super(ParentNetwork, self).__init__()

      self.layer_list = np.array([0] + layer_list)
      self.additive_activation_ratio = np.zeros(len(layer_list))
      self.additive_in_activation_ratio = np.zeros(len(layer_list))

      self.first_layer = nn.Sequential(nn.Linear(n_in, layer_list[0]))
      self.hidden_layers = nn.Sequential()
      for i in range(1, len(layer_list)):
          self.hidden_layers.add_module(f"linear_{i}", nn.Linear(layer_list[i - 1], layer_list[i]))
          # if i != len(layer_list) - 1:
          #     self.hidden_layers.add_module(f"relu_{i}", nn.ReLU())

      self.apply(self.init_weights)
      self.output_dim = layer_list[-1]
      # This should be a 0/1 array showing that for a given datapoint.
      self.neurons_activations = np.zeros(np.sum(layer_list))
      # This is an array showing each neuron is activated for how many datapoints before reseting.
      self.additive_activations = np.zeros(np.sum(layer_list))
      

    def get_layer_list(self):
       return self.layer_list[1:]
    
    @abstractmethod
    def forward(self):
       pass
    
    def reset_all(self):
       self.reset()
       self.reset_neurons_activation()
       
    def reset(self, return_pre_activations=False, return_activation_values=False, single_point=False):
      '''
      Resets activation tracking arrays.
      '''
      self.additive_activation_ratio = np.zeros(len(self.get_layer_list()))
      self.additive_in_activation_ratio = np.zeros(len(self.get_layer_list()))

      self.neurons_activations = np.zeros(np.sum(self.get_layer_list()))
      self.additive_activations = np.zeros(np.sum(self.get_layer_list()))
    
    def reset_neurons_activation(self):
       self.neurons_activations = np.zeros(np.sum(self.get_layer_list()))

    def analysis_neurons_layer_wise_animation(self, cumulative_activations, num):
      layer_activation_ratio = []
      i = 0
      layer_activation_ratio.append([cumulative_activations[self.layer_list[i]: self.layer_list[i + 1]] / num])
      i += 1
      for layer in self.hidden_layers:
        layer_activation_ratio.append([cumulative_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] / num])
        i += 1
      # inactive_neurons = self.get_layer_list() - layer_activation_ratio
      # self.additive_activation_ratio += layer_activation_ratio / self.get_layer_list()
      # self.additive_in_activation_ratio += inactive_neurons / self.get_layer_list()
      return layer_activation_ratio

    def analysis_neurons_activations_depth_wise(self, num):
      '''
        Analyzes and calculates the activation ratio depth-wise in the network.

        Returns:
        Array of activation ratios for each layer.
      '''
      layer_activation_ratio = np.zeros(self.get_layer_list().shape[0])
      i = 0
      layer_activation_ratio[i] = np.sum(self.additive_activations[self.layer_list[i]: self.layer_list[i + 1]]) / num
      i += 1
      for layer in self.hidden_layers:
        layer_activation_ratio[i] = np.sum(self.additive_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]]) / num
        i += 1
      inactive_neurons = self.get_layer_list() - layer_activation_ratio
      self.additive_activation_ratio += layer_activation_ratio / self.get_layer_list()
      self.additive_in_activation_ratio += inactive_neurons / self.get_layer_list()

      return layer_activation_ratio

    def init_weights(self, m, init_type='he'):
        if isinstance(m, nn.Linear):
            if init_type == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=1)
            # Initialize biases to small values uniformly
            nn.init.uniform_(m.bias, -0.001, 0.001)

    def get_all_parameters(self):
        weights = []
        biases = []
        for k, v in self.state_dict().items():
            if 'weight' in k:
                weights.append(v.T)
            if 'bias' in k:
                biases.append(v)
        return weights, biases
    
    @abstractmethod
    def other_forward_analysis(self, data):
       pass
    
class MLP_ReLU(ParentNetwork):
    '''
    This class implements a multi-layer perceptron (MLP) with ReLU (Rectified Linear Unit) activations. 
    It is designed for flexibility in defining the network architecture 
    through a specified number of input dimensions and a list of layer sizes.
    '''

    def __init__(self, n_in, layer_list):
        super(MLP_ReLU, self).__init__(n_in, layer_list)

    def better_forwad(self, x):
        def whole_dataset(this_data):
            res_list = count_near_zero_eigenvalues(this_data, return_eigenvalues=False)
            eigenvalues_count.append(res_list)
            # distances_this_data = c(this_data)
            distances_this_data = cdist(this_data, this_data)[np.triu_indices(this_data.shape[0], k=1)]
            dis_values.append(distances_this_data / np.mean(distances_this_data))
            dis_stats.append([np.mean(distances_this_data), np.max(distances_this_data), np.min(distances_this_data)])

        eigenvalues_count = []
        dis_values = []
        dis_stats = []
        whole_dataset(x)

        new_pre_activation = self.first_layer(x)
        new_activation = nn.ReLU()(new_pre_activation)
        i = 0
        deteached_version = new_activation.cpu().clone().detach().numpy()
        self.neurons_activations[np.sum(self.layer_list[:i + 1]): \
                                 np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] =\
                                np.sum((deteached_version > 0), axis=0)
        i += 1
        whole_dataset(deteached_version)

        for layer in self.hidden_layers:
            new_pre_activation = layer(new_activation)
            new_activation = nn.ReLU()(new_pre_activation)
            deteached_version = new_activation.cpu().clone().detach().numpy()

            self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) \
                                     + self.layer_list[i + 1]] =\
                                    np.sum((deteached_version > 0), axis=0)
            i += 1
            whole_dataset(deteached_version)
        


        self.additive_activations += self.neurons_activations
        self.reset_neurons_activation()
        return eigenvalues_count, np.array(dis_values), dis_stats

    
    def forward(self, x, data=None, device=None, return_pre_activations=False, return_activation_values=False, single_point=False):
        '''
        Performs a forward pass of the network.

        Parameters:
        x (Tensor): Input tensor.
        return_pre_activations (bool): If True, returns pre-activation values.
        return_activation_values (bool): If True, returns activation values.
        single_point (bool): Flag used for single data point processing.

        Returns:
        Output of the network, and optionally, pre-activations and activations.
        '''
        first_layer_result = self.first_layer(x)
        output = self.hidden_layers(first_layer_result)
        i = 0

        if return_pre_activations:
          new_pre_activation = self.first_layer(x)
          # Apply ReLU activation to the first layer output
          new_activation = nn.ReLU()(new_pre_activation)
          # new_activation = nn.LeakyReLU()(new_pre_activation)
          # new_activation = new_pre_activation.clone()

          if return_activation_values:
            pre_activations = [new_pre_activation]
            activations = [new_activation]

          i = 0
          self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = (new_activation.cpu().clone().detach() > 0).numpy()
          i += 1
          # Process hidden layers
          for layer in self.hidden_layers:
              new_pre_activation = layer(new_activation)
              # Apply ReLU, except for the output layer
              new_activation = nn.ReLU()(new_pre_activation)
              # new_activation = new_pre_activation.clone()
              # new_activation = nn.LeakyReLU()(new_pre_activation)

              if return_activation_values:
                pre_activations.append(new_pre_activation)
                activations.append(new_activation)
              self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = (new_activation.cpu().clone().detach() > 0).numpy()
              i += 1
              ########## TEST the whole procedure later
          if not single_point:
                self.additive_activations += self.neurons_activations
                self.reset_neurons_activation()
          if return_activation_values:
            return output, pre_activations, activations
        return output
        
    def other_forward_analysis(self, data, device, rep=False,return_eigenvalues=False, plot_projection=True, calculate_distance=True, projection_analysis_bool=False):
        eigenvalues = []
        eigenvalues_count = []
        plot_list_pca_2d = []
        plot_list_pca_3d = []
        plot_list_random_2d = []
        plot_list_random_3d = []
        dis_values = []
        dis_stats = []

        def append_handling(this_data):
          res_list = count_near_zero_eigenvalues(this_data, return_eigenvalues=return_eigenvalues)
          if return_eigenvalues:
            eigenvalues_count.append(res_list[0])
            eigenvalues.append(res_list[1])
          else:
            eigenvalues_count.append(res_list)
          if not rep:
            if projection_analysis_bool:
              if plot_projection:
                plot_list_pca_2d.append(projection_analysis(this_data, 'pca', 2))
                plot_list_pca_3d.append(projection_analysis(this_data, 'pca', 3))
                plot_list_random_2d.append(projection_analysis(this_data, 'random', 2))
                plot_list_random_3d.append(projection_analysis(this_data, 'random', 3))
                distances_this_data = c(this_data)
                dis_values.append(distances_this_data / np.mean(distances_this_data))
                dis_stats.append([np.mean(distances_this_data), np.max(distances_this_data), np.min(distances_this_data)])
          else:
            distances_this_data = c(this_data)
            dis_values.append(distances_this_data / np.mean(distances_this_data))
            dis_stats.append([np.mean(distances_this_data), np.max(distances_this_data), np.min(distances_this_data)])

        append_handling(data)

        new_pre_activation = self.first_layer(torch.tensor(data, dtype=torch.float32).to(device))
        new_activation = nn.ReLU()(new_pre_activation)
        # new_activation = new_pre_activation.clone()
        # new_activation = nn.LeakyReLU()(new_pre_activation)
        new_data = new_activation.detach().clone().detach().numpy()
        append_handling(new_data)

        # Process hidden layers
        for layer in self.hidden_layers:
            new_pre_activation = layer(new_activation)
            new_activation = nn.ReLU()(new_pre_activation)
            # new_activation = new_pre_activation.clone()
            # new_activation = nn.LeakyReLU()(new_pre_activation)
            new_data = new_activation.detach().clone().detach().numpy()

            append_handling(new_data)
        if rep:
           return eigenvalues_count, dis_values, dis_stats
        if projection_analysis_bool:
          if return_eigenvalues:
            return eigenvalues_count, eigenvalues, plot_list_pca_2d, plot_list_pca_3d, \
            plot_list_random_2d, plot_list_random_3d, dis_values, dis_stats
          else:
            return eigenvalues_count, plot_list_pca_2d, plot_list_pca_3d, plot_list_random_2d, \
              plot_list_random_3d, dis_values, dis_stats
        else:
          if return_eigenvalues:
            return eigenvalues_count, eigenvalues
          else:
            return eigenvalues_count

class ResNet_arch(nn.Module):
   
  def __init__(self, n_in, layer_list):
      super(ResNet_arch, self).__init__(n_in, layer_list)

      # Residual connection
      if self.input_dim != self.output_dim:
          self.residual_layer = nn.Linear(n_in, self.output_dim)
      else:
          self.residual_layer = None

  def forward(self, x, return_pre_activations=False, return_activation_values=False, single_point=False):
      first_layer_result = self.first_layer(x)
      output = self.hidden_layers(first_layer_result)
      i = 0

      if return_pre_activations:
        new_pre_activation = self.first_layer(x)
        # Apply ReLU activation to the first layer output
        new_activation = nn.ReLU()(new_pre_activation)

        if return_activation_values:
          pre_activations = [new_pre_activation]
          activations = [new_activation]

        i = 0
        self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = (new_activation.clone().detach() > 0).numpy()
        i += 1

        # Process hidden layers
        for layer in self.hidden_layers:
            new_pre_activation = layer(new_activation)
            # Apply ReLU, except for the output layer
            new_activation = nn.ReLU()(new_pre_activation)

            if return_activation_values:
              pre_activations.append(new_pre_activation)
              activations.append(new_activation)

            self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = (new_activation.clone().detach() > 0).numpy()
            i += 1
            ########## TEST the whole procedure later
        if not single_point:
              self.additive_activations += self.neurons_activations
              self.reset_neurons_activation()
        if return_activation_values:
          return output, pre_activations, activations
      return output
     
     
   
   