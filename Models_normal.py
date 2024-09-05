from utils import *
from utils import nn
from utils import np



class MLP_simple(nn.Module):
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
    def __init__(self, bias=0, init_scale=1):
        super(MNIST_classifier, self).__init__()

        self.bias = bias
        self.init_scale = init_scale

        self.flatten = nn.Flatten()
        self.layers = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

        self.init_all_weights()

    def forward(self, x):
        x = self.flatten(x)
        x = self.layers(x)
        return x
    
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

