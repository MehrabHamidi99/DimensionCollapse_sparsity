from utils import *
from utils import nn
from utils import np
import torch.nn.functional as F


# class MLP_simple(nn.Module):
#     def __init__(self, n_in, layer_list, bias=1e-4, init_scale=1, activation=nn.ReLU):
#         super(MLP_simple, self).__init__()

#         self.n_in = n_in

#         self.bias = bias
#         self.init_scale = init_scale

#         self.layer_list = layer_list
        
#         self.layers = nn.Sequential()

#         self.activation = activation

#         self.layers.add_module(f"linear_{0}", nn.Linear(n_in, layer_list[0]))
#         self.layers.add_module(f"relu_{0}", self.activation())
#         for i in range(len(layer_list) - 1):
#             self.layers.add_module(f"linear_{i + 1}", nn.Linear(layer_list[i], layer_list[i + 1]))
#             self.layers.add_module(f"relu_{i + 1}", self.activation())

#         self.init_all_weights()

#     def forward(self, x):
#         output = self.layers(x)
#         return output
    
#     def init_all_weights(self, device='cpu'):
#             self.apply(lambda m: self.init_weights(m))
#             self.to(device)

#     def init_weights(self, m, init_type='he'):
#         if isinstance(m, nn.Linear):
#             if init_type == 'he':
#                 # Reinitialize weights using He initialization (scaled by init_scale)
#                 m.weight.data.normal_(0, torch.sqrt(torch.tensor(2.0 / m.weight.size(1)))) # type: ignore
#                 # m.weight.data *= self.init_scale / m.weight.norm(
#                 #     dim=tuple(range(1, m.weight.data.ndim)), p=2, keepdim=True
#                 # )
#             elif init_type == 'xavier':
#                 nn.init.xavier_normal_(m.weight, gain=1)

#             if m.bias is not None:
#                 m.bias.data.fill_(self.bias)

class MLP_simple(nn.Module):
    def __init__(self, n_in, layer_list, bias=1e-4, init_scale=1, activation=nn.ReLU, use_batch_norm=True):
        super(MLP_simple, self).__init__()

        self.n_in = n_in
        self.bias = bias
        self.init_scale = init_scale
        self.layer_list = layer_list
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        
        self.layers = nn.Sequential()

        # Input layer
        self.layers.add_module(f"linear_0", nn.Linear(n_in, layer_list[0]))
        if self.use_batch_norm:
            self.layers.add_module(f"bn_0", nn.BatchNorm1d(layer_list[0]))
        self.layers.add_module(f"activation_0", self.activation())

        # Hidden layers
        for i in range(len(layer_list) - 1):
            self.layers.add_module(f"linear_{i+1}", nn.Linear(layer_list[i], layer_list[i+1]))
            if self.use_batch_norm:
                self.layers.add_module(f"bn_{i+1}", nn.BatchNorm1d(layer_list[i+1]))
            self.layers.add_module(f"activation_{i+1}", self.activation())

        self.init_all_weights()

    def forward(self, x):
        return self.layers(x)
    
    def init_all_weights(self, device='cpu'):
        self.apply(lambda m: self.init_weights(m))
        self.to(device)

    def init_weights(self, m, init_type='he'):
        if isinstance(m, nn.Linear):
            if init_type == 'he':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=1)

            if m.bias is not None:
                m.bias.data.fill_(self.bias)

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


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
    
    def init_all_weights(self, device='cpu'):
            self.apply(lambda m: self.init_weights(m))
            self.to(device)

    def init_weights(self, m, init_type='he'):
        if isinstance(m, nn.Linear):
            if init_type == 'he':
                # Reinitialize weights using He initialization (scaled by init_scale)
                m.weight.data.normal_(0, torch.sqrt(torch.tensor(2.0 / m.weight.size(1)))) # type: ignore
                # m.weight.data *= self.init_scale / m.weight.norm(
                #     dim=tuple(range(1, m.weight.data.ndim)), p=2, keepdim=True
                # )
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=1)

            if m.bias is not None:
                m.bias.data.fill_(self.bias)




class CIFAR_Res_classifier(nn.Module):
    def __init__(self, n_in=784, layer_list=[256, 128, 64, 32, 10], bias=0.0, init_scale=1):
        super(CIFAR_Res_classifier, self).__init__()

        self.n_in = n_in

        self.bias = bias
        self.init_scale = init_scale

        self.layer_list = layer_list

        self.layers = nn.Sequential()

        self.layers.add_module(f"linear_{0}", nn.Linear(n_in, layer_list[0]))
        # self.layers.add_module(f"relu_{0}", nn.ReLU())
        for i in range(len(layer_list) - 1):
            self.layers.add_module(f"linear_{i + 1}", nn.Linear(layer_list[i], layer_list[i + 1]))
            # if i < len(layer_list) - 2:
            #     self.layers.add_module(f"relu_{i + 1}", nn.ReLU())
        
        self.log_softmax = nn.LogSoftmax(dim=1)


        self.init_all_weights()

    def forward(self, x):
        x = x.view(-1, self.n_in)
        residual = torch.tensor([0.0], dtype=x.dtype).requires_grad_(True).to(x.device)

        for i, layer in enumerate(self.layers):
            out = layer(x)
            # Add skip connection for every 5 layers
            if i % 5 == 0 and i != 0:
                out = out + residual  # Avoid in-place addition
                residual = out.detach()  # Detach to avoid modifying tensor that requires gradient
            
            if i != len(self.layers) - 1:  # Apply ReLU to all except the last layer
                out = F.relu(out, inplace=False)  # Avoid in-place operation by setting inplace=False
                if i % 5 == 0 and i != 0:
                    residual = out.detach()  # Detach to avoid modifying tensor that requires gradient

            x = out

        output = self.log_softmax(x)
        return output
    
    def init_all_weights(self, device='cpu'):
            self.apply(lambda m: self.init_weights(m))
            self.to(device)

    def init_weights(self, m, init_type='he'):
        if isinstance(m, nn.Linear):
            if init_type == 'he':
                # Reinitialize weights using He initialization (scaled by init_scale)
                m.weight.data.normal_(0, torch.sqrt(torch.tensor(2.0 / m.weight.size(1)))) # type: ignore
                # m.weight.data *= self.init_scale / m.weight.norm(
                #     dim=tuple(range(1, m.weight.data.ndim)), p=2, keepdim=True
                # )
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=1)

            if m.bias is not None:
                m.bias.data.fill_(self.bias)

