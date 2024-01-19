from utils import *
from abc import ABC, abstractmethod
from utils import nn
from utils import np


class CustomLinearWithActivation(nn.Linear):
    def __init__(self, in_features, out_features, activation, bias=True, additional_analysis=True):
        super(CustomLinearWithActivation, self).__init__(in_features, out_features, bias)
        self.activation = activation
        self.additional_analysis = additional_analysis
        self.extra = False
        self.reinitialize_storing_values()

        
    def reinitialize_storing_values(self):
        self.non_zero = 0
        if self.additional_analysis:
          self.eigenvalues_count = []
          self.dis_values = []
          self.dis_stats = []

        if self.extra:
          self.eigenvalues = []
          self.plot_list_pca_2d = []
          self.plot_list_pca_3d = []
          self.plot_list_random_2d = []
          self.plot_list_random_3d = []

    def forward(self, input):
        # Call the original forward method to get the linear transformation
        output = super(CustomLinearWithActivation, self).forward(input)
        output = self.activation(output)

        deteached_version = output.cpu().clone().detach().numpy()
        self.non_zero += np.sum((deteached_version > 0), axis=0)
        if self.extra:
          self.eigenvalues, self.plot_list_pca_2d, self.plot_list_pca_3d, self.plot_list_random_2d, self.plot_list_random_3d = projection_analysis_for_full_data(deteached_version, True)
        if self.additional_analysis:
          self.eigenvalues_count, self.dis_values, self.dis_stats = additional_analysis_for_full_data(deteached_version)
        return output

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
    additive_activations (np.array): Array to accumulate activations over inputs or epochs.


    Usage Example

    model = MLP_ReLU(n_in=10, layer_list=[20, 30, 40])
    output = model(torch.randn(10))

    '''
    def __init__(self, n_in, layer_list, bias=1e-3, additional_analysis=True):
      super(ParentNetwork, self).__init__()

      self.BIAS = bias

      self.layer_list = np.array([0] + layer_list)
      self.layers = nn.Sequential()
      i = 0
      self.layers.add_module(f"linear_{i}", CustomLinearWithActivation(n_in, layer_list[i], nn.ReLU(), additional_analysis=additional_analysis))
      for i in range(1, len(layer_list)):
          self.layers.add_module(f"linear_{i}", CustomLinearWithActivation(layer_list[i - 1], layer_list[i], nn.ReLU(), additional_analysis=additional_analysis))
      self.apply(self.init_weights)
      self.output_dim = layer_list[-1]
      # This is an array showing each neuron is activated for how many datapoints before reseting.
      self.additive_activations = np.zeros(np.sum(layer_list))
      self.additional_analysis = additional_analysis

      self.extra_mode = False

      self.change_key = True
      
    def get_layer_list(self):
       return self.layer_list[1:]
    
    def extra(self):
      '''
      Should call it before post_forward_neuron_activation_analysis:
      extra()
      forward()
      post_forward_neuron_activation_analysis()
      '''
      if not self.change_key:
         raise Exception("Try to change mode while change key is false!")
      self.extra_mode = True
      for layer in self.layers:
          layer.extra = self.extra_mode

    def not_extra(self):
      if not self.change_key:
         raise Exception("Try to change mode while change key is false!")
      self.extra_mode = False
      for layer in self.layers:
          layer.extra = self.extra_mode
       
    def reset(self):
      '''
      Resets activation tracking arrays.
      '''
      self.additive_activations = np.zeros(np.sum(self.get_layer_list()))
      for layer in self.layers:
         layer.reinitialize_storing_values()

    def analysis_neurons_layer_wise_animation(self, cumulative_activations, num):
      layer_activation_ratio = []
      i = 0
      for _ in self.layers:
        layer_activation_ratio.append([cumulative_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] / num])
        i += 1
      return layer_activation_ratio

    def init_weights(self, m, init_type='he'):
        if isinstance(m, nn.Linear):
            if init_type == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=1)
            # Initialize biases to small values uniformly
            nn.init.uniform_(m.bias, -1 * self.BIAS, self.BIAS)

    def get_all_parameters(self):
        weights = []
        biases = []
        for k, v in self.state_dict().items():
            if 'weight' in k:
                weights.append(v.T)
            if 'bias' in k:
                biases.append(v)
        return weights, biases
    
    def post_forward_neuron_activation_analysis(self, full_data=None):
      if full_data is not None:
        if torch.is_tensor(full_data):
            full_data = full_data.cpu().clone().detach().numpy()
        if self.extra_mode:
          tmp_res = projection_analysis_for_full_data(full_data, True)
          eigenvalues = [tmp_res[0]]
          plot_list_pca_2d = [tmp_res[1]]
          plot_list_pca_3d = [tmp_res[2]]
          plot_list_random_2d = [tmp_res[3]]
          plot_list_random_3d = [tmp_res[4]]   
        if self.additional_analysis:
          tmp_res = additional_analysis_for_full_data(full_data)
          eigenvalues_count = [tmp_res[0]]
          dis_values = [np.array(tmp_res[1])]
          dis_stats = [tmp_res[2]]
      else:
        if self.extra_mode:
          eigenvalues = []
          plot_list_pca_2d = []
          plot_list_pca_3d = []
          plot_list_random_2d = []
          plot_list_random_3d = []   
        if self.additional_analysis:
          eigenvalues_count = []
          dis_values = []
          dis_stats = []
         
      i = 0
  
      for layer in self.layers:
        self.additive_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = layer.non_zero
        if self.extra_mode:
          eigenvalues.append(layer.eigenvalues)
          plot_list_pca_2d.append(layer.plot_list_pca_2d)
          plot_list_pca_3d.append(layer.plot_list_pca_3d)
          plot_list_random_2d.append(layer.plot_list_random_2d)
          plot_list_random_3d.append(layer.plot_list_random_3d)
        if self.additional_analysis:
          eigenvalues_count.append(layer.eigenvalues_count)
          dis_values.append(np.array(layer.dis_values))
          dis_stats.append(layer.dis_stats)
        i += 1

      self.change_key = True
      
      if self.extra_mode:
        return self.additive_activations, eigenvalues_count, eigenvalues, plot_list_pca_2d, plot_list_pca_3d, plot_list_random_2d, plot_list_random_3d, np.array(dis_values), dis_stats
      if self.additional_analysis:
        return self.additive_activations, eigenvalues_count, np.array(dis_values, dtype=object), dis_stats
      else:
        return self.additive_activations
      
    def analysis_neurons_activations_depth_wise(self, num):
      ##### TODO!!!!!
      '''
        Analyzes and calculates the activation ratio.

        Returns:
        Array of activation ratios for each layer.
      '''

      layer_activation_ratio = np.zeros(self.get_layer_list().shape[0])
      i = 0
      for _ in self.layers:
        layer_activation_ratio[i] = np.sum(self.additive_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]]) / num
        i += 1
      inactive_neurons = self.get_layer_list() - layer_activation_ratio
      self.additive_activation_ratio += layer_activation_ratio / self.get_layer_list()
      self.additive_in_activation_ratio += inactive_neurons / self.get_layer_list()

      return layer_activation_ratio
   