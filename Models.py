from utils import *
from abc import ABC, abstractmethod
from utils import nn
from utils import np
from ParentModels import *


class MLP_mnist(ParentNetwork):

  def __init__(self, hiddens=[128, 64], bias=1e-4):
     super(MLP_mnist, self).__init__(n_in=784, layer_list= hiddens + [10], bias=bias)

  def forward(self, x):
      # flatten_x = torch.flatten(x, start_dim=1)
      output = self.layers(x.flatten(1))
      return output

class MLP_ReLU(ParentNetwork):
    '''
    This class implements a multi-layer perceptron (MLP) with ReLU (Rectified Linear Unit) activations. 
    It is designed for flexibility in defining the network architecture 
    through a specified number of input dimensions and a list of layer sizes.
    '''

    def __init__(self, n_in, layer_list, bias=1e-4):
        super(MLP_ReLU, self).__init__(n_in, layer_list, bias=bias)

    def forward(self, x):
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
        output = self.layers(x)
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
     
     
   
   