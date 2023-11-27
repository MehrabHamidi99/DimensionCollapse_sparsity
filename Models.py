from utils import *
from abc import ABC, abstractmethod
from utils import nn
from utils import np


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

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.xavier_normal_(torch.tensor(m.weight, dtype=torch.double))
            # torch.nn.init.xavier_normal_(torch.tensor(m.weight))
            nn.init.xavier_normal_(m.weight, gain=1)
            # m.bias.data.fill_(torch.tensor(np.random.randn() + 0.1, dtype=torch.double))
            # m.bias.data.fill_(torch.tensor(np.random.randn() + 0.1))
            m.bias = nn.Parameter(m.bias / 100)

    def get_all_parameters(self):
        weights = []
        biases = []
        for k, v in self.state_dict().items():
            if 'weight' in k:
                weights.append(v.T)
            if 'bias' in k:
                biases.append(v)
        return weights, biases
    
class MLP_ReLU(ParentNetwork):
    '''
    This class implements a multi-layer perceptron (MLP) with ReLU (Rectified Linear Unit) activations. 
    It is designed for flexibility in defining the network architecture 
    through a specified number of input dimensions and a list of layer sizes.
    '''

    def __init__(self, n_in, layer_list):
        super(MLP_ReLU, self).__init__(n_in, layer_list)
    
    def forward(self, x, return_pre_activations=False, return_activation_values=False, single_point=False):
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

          if return_activation_values:
            pre_activations = [new_pre_activation]
            activations = [new_activation]

          i = 0
          self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = (new_activation.clone().detach() > 0).long().numpy()
          i += 1

          # Process hidden layers
          for layer in self.hidden_layers:
              new_pre_activation = layer(new_activation)
              # Apply ReLU, except for the output layer
              new_activation = nn.ReLU()(new_pre_activation)

              if return_activation_values:
                pre_activations.append(new_pre_activation)
                activations.append(new_activation)

              self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = (new_activation.clone().detach() > 0).long().numpy()
              i += 1
              ########## TEST the whole procedure later
          if not single_point:
                self.additive_activations += self.neurons_activations
                self.reset_neurons_activation()
          if return_activation_values:
            return output, pre_activations, activations
        return output


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
        self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = (new_activation.clone().detach() > 0).long().numpy()
        i += 1

        # Process hidden layers
        for layer in self.hidden_layers:
            new_pre_activation = layer(new_activation)
            # Apply ReLU, except for the output layer
            new_activation = nn.ReLU()(new_pre_activation)

            if return_activation_values:
              pre_activations.append(new_pre_activation)
              activations.append(new_activation)

            self.neurons_activations[np.sum(self.layer_list[:i + 1]): np.sum(self.layer_list[:i + 1]) + self.layer_list[i + 1]] = (new_activation.clone().detach() > 0).long().numpy()
            i += 1
            ########## TEST the whole procedure later
        if not single_point:
              self.additive_activations += self.neurons_activations
              self.reset_neurons_activation()
        if return_activation_values:
          return output, pre_activations, activations
      return output
     
     
   
   