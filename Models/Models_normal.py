from utils import *
from utils import nn
from utils import np
import torch.nn.functional as F


class MLP_simple(nn.Module):
    def __init__(self, n_in, layer_list, bias=1e-4, init_scale=1, activation=nn.ReLU):
        super(MLP_simple, self).__init__()

        self.n_in = n_in

        self.bias = bias
        self.init_scale = init_scale

        self.layer_list = layer_list
        
        self.layers = nn.Sequential()

        self.activation = activation

        self.layers.add_module(f"linear_{0}", nn.Linear(n_in, layer_list[0]))
        self.layers.add_module(f"relu_{0}", self.activation())
        for i in range(len(layer_list) - 1):
            self.layers.add_module(f"linear_{i + 1}", nn.Linear(layer_list[i], layer_list[i + 1]))
            self.layers.add_module(f"relu_{i + 1}", self.activation())

        self.init_all_weights()

    def forward(self, x):
        output = self.layers(x)
        return output
    
    def init_all_weights(self):
        self.apply(self.init_weights)

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


class MNIST_classifier(nn.Module):
    def __init__(self, n_in=784, layer_list=[256, 128, 64, 32, 10], bias=0.0, init_scale=1):
        super(MNIST_classifier, self).__init__()

        self.n_in = n_in

        self.bias = bias
        self.init_scale = init_scale

        self.layer_list = layer_list

        self.layers = nn.Sequential()

        self.layers.add_module(f"linear_{0}", nn.Linear(n_in, layer_list[0]))
        self.layers.add_module(f"relu_{0}", nn.ReLU())
        for i in range(len(layer_list) - 1):
            self.layers.add_module(f"linear_{i + 1}", nn.Linear(layer_list[i], layer_list[i + 1]))
            if i < len(layer_list) - 2:
                self.layers.add_module(f"relu_{i + 1}", nn.ReLU())
        
        self.log_softmax = nn.LogSoftmax(dim=1)


        self.init_all_weights()

    def forward(self, x):
        output = self.layers(x.view(-1, self.n_in))
        output = self.log_softmax(output)
        return output
    
    def init_all_weights(self):
        self.apply(self.init_weights)

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



class CIFAR_Res_classifier(nn.Module):
    def __init__(self, n_in=784, layer_list=[256, 128, 64, 32, 10], bias=0.0, init_scale=1):
        super(CIFAR_Res_classifier, self).__init__()

        self.n_in = n_in

        self.bias = bias
        self.init_scale = init_scale

        self.layer_list = layer_list

        self.layers = nn.Sequential()

        self.layers.add_module(f"linear_{0}", nn.Linear(n_in, layer_list[0]))
        self.layers.add_module(f"relu_{0}", nn.ReLU())
        for i in range(len(layer_list) - 1):
            self.layers.add_module(f"linear_{i + 1}", nn.Linear(layer_list[i], layer_list[i + 1]))
            if i < len(layer_list) - 2:
                self.layers.add_module(f"relu_{i + 1}", nn.ReLU())
        
        self.log_softmax = nn.LogSoftmax(dim=1)


        self.init_all_weights()

    def forward(self, x):
        x = x.view(-1, self.n_in)
        residual = 0
        
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Add skip connection for every 5 layers
            if (i + 1) % 5 == 0 and i != len(self.layers) - 1:
                x += residual
                residual = x

        output = self.log_softmax(x)
        return output
    
    def init_all_weights(self):
        self.apply(self.init_weights)

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