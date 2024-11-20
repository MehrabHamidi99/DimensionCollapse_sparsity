from utils import *
from abc import ABC, abstractmethod
from utils import nn
from utils import np
from Models.ParentModels import *


class MLP_mnist(ParentNetwork):

  def __init__(self, hiddens=[128, 64], bias=1e-4):
     super(MLP_mnist, self).__init__(n_in=784, layer_list= hiddens + [10], bias=bias)

  def forward(self, x):
      # flatten_x = torch.flatten(x, start_dim=1)
      self.change_key = False
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
        self.change_key = False
        output = self.layers(x)
        return output


class ResNet_one_skip_connection(ParentNetwork):
    '''
    This class implements a multi-layer perceptron (MLP) with ReLU (Rectified Linear Unit) activations. 
    It is designed for flexibility in defining the network architecture 
    through a specified number of input dimensions and a list of layer sizes.
    '''

    def __init__(self, n_in, layer_list, bias=1e-4):
        super(ResNet_one_skip_connection, self).__init__(n_in, layer_list, bias=bias)
        self.layer_list = np.array([0] + layer_list)
        self.middle_layers = nn.Sequential()
        i = 0
        self.first_layer = CustomLinearWithActivation(n_in, layer_list[i], nn.ReLU(), additional_analysis=True)
        for i in range(1, len(layer_list) - 1):
            self.middle_layers.add_module(f"linear_{i}", CustomLinearWithActivation(layer_list[i - 1], layer_list[i], nn.ReLU(), additional_analysis=True))
        i += 1
        self.last_layer = CustomLinearWithActivation(layer_list[i - 1], layer_list[i], nn.ReLU(), additional_analysis=True)
        self.setup(layer_list=layer_list)

        self.layers = nn.Sequential()
        self.layers.add_module(f"linear_{0}", self.first_layer)
        i = 0
        for layer in self.middle_layers:
           self.layers.add_module(f"hidden_{i}", layer)
           i += 1
        self.layers.add_module(f"linear_{i}", self.last_layer)

        self.setup(layer_list=layer_list)

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
        self.change_key = False
        out1 = self.first_layer(x)
        out2 = self.middle_layers(out1)
        output = self.last_layer(out2 + x)
        return output

class ResNet_arch(ParentNetwork):
   
  def __init__(self, n_in, layer_list, bias=1e-4):
    super(ResNet_arch, self).__init__(n_in, layer_list, bias=bias)

    self.layer_list = np.array([0] + layer_list)
    self.layers = nn.Sequential()
    i = 0
    this_layer = ResidualBuildingBlock(n_in, layer_list[i], nn.ReLU())
    cur_layer = this_layer
    self.layers.add_module(f"linear_{i}", this_layer)
    for i in range(1, len(layer_list)):
        this_layer = ResidualBuildingBlock(layer_list[i - 1], layer_list[i], nn.ReLU())
        if i % 3 == 0:
           cur_layer.out_res_layer = this_layer
        self.layers.add_module(f"linear_{i}", this_layer)
        cur_layer = this_layer
    self.setup(layer_list=layer_list)

  def forward(self, x):
      self.change_key = False
      output = self.layers(x)
      return output
     
     
   
   