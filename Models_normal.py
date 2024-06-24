from utils import *
from utils import nn
from utils import np



class MLP_simple(nn.Module):
    '''
    This class implements a multi-layer perceptron (MLP) with ReLU (Rectified Linear Unit) activations. 
    It is designed for flexibility in defining the network architecture 
    through a specified number of input dimensions and a list of layer sizes.
    '''

    def __init__(self, n_in, layer_list, bias=0, init_scale=1):
        super(MLP_simple, self).__init__()

        self.bias = bias
        self.init_scale = init_scale
        
        self.layers = nn.Sequential()

        self.layers.add_module(f"linear_{0}", nn.Linear(n_in, layer_list[0]))
        self.layers.add_module(f"relu_{0}", nn.ReLU())
        for i in range(len(layer_list) - 1):
            self.layers.add_module(f"linear_{i + 1}", nn.Linear(layer_list[i], layer_list[i + 1]))
            self.layers.add_module(f"relu_{i + 1}", nn.ReLU())

        self.apply(self.init_weights)

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
    
    def init_weights(self, m, init_type='he'):
        if isinstance(m, nn.Linear):
            if init_type == 'he':
                # Init Weights (Fan-In)
                # stdv = 1. / torch.sqrt(m.weight.size(1))
                # m.weight.data.normal_(0, stdv)
                m.weight.data *= self.init_scale / m.weight.norm(
                    dim=tuple(range(1, m.weight.data.ndim)), p=2, keepdim=True
                )
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=1)

            m.bias.data *= self.bias