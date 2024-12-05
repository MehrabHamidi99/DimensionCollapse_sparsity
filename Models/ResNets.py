import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetBlock(nn.Module):
    def __init__(self, in_features, out_features, res_net_dim=64, depth=3):
        super(ResNetBlock, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.res_net_dim = res_net_dim
        self.depth = depth

        # Defining the first linear layer
        self.linear1 = nn.Linear(in_features, res_net_dim)
        
        # Defining the internal layers with a loop
        self.internal_layers = nn.Sequential()
        for i in range(depth - 2):  # Loop to create intermediate linear layers
            self.internal_layers.add_module(f"linear_{i+2}", nn.Linear(res_net_dim, res_net_dim))
            self.internal_layers.add_module(f"relu_{i+2}", nn.ReLU())

        # Defining the final linear layer
        self.last_layer = nn.Sequential(
            nn.Linear(res_net_dim, out_features),
            nn.ReLU(),
            nn.BatchNorm1d(out_features)
        )
        
    def forward(self, x):
        # Initial computation with input features
        residual = self.linear1(x)
        
        # Forward pass through internal layers
        out = residual
        out = self.internal_layers(out)
        
        # Adding the residual connection after processing through internal layers
        out = out + residual
        
        # Forward pass through the last layer and batch normalization
        out = self.last_layer(out)
        
        return out


class CIFAR_Res_classifier(nn.Module):
    def __init__(self, n_in=784, layer_list=[256, 128, 64, 32, 10], bias=0.0, init_scale=1, res_net_dim=64, res_block_depth=3):
        super(CIFAR_Res_classifier, self).__init__()

        self.n_in = n_in
        self.bias = bias
        self.init_scale = init_scale
        self.layer_list = layer_list
        self.res_net_dim = res_net_dim
        self.res_block_depth = res_block_depth

        self.layers = nn.Sequential()
        
        # Initial Linear Layer
        self.layers.add_module(f"linear_{0}", nn.Linear(n_in, layer_list[0]))
        self.layers.add_module(f"relu_{0}", nn.ReLU())

        # Adding ResNet blocks
        for i in range(1, len(layer_list)):
            in_features = layer_list[i - 1]
            out_features = layer_list[i]
            self.layers.add_module(f"resnet_block_{i}", ResNetBlock(in_features, out_features, out_features, depth=res_block_depth))
        
        # Add a final linear layer before output, matching the output size
        self.layers.add_module("last_layer_classifier", nn.Linear(self.layer_list[-1], self.layer_list[-1]))

        # LogSoftmax layer
        self.log_softmax = nn.LogSoftmax(dim=1)

        # Initialize weights
        self.init_all_weights()

    def forward(self, x):
        x = x.view(-1, self.n_in)  # Flatten the input

        # Forward pass through all layers
        x = self.layers(x)
        
        # Apply LogSoftmax
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
                m.weight.data *= self.init_scale / m.weight.norm(
                    dim=tuple(range(1, m.weight.data.ndim)), p=2, keepdim=True
                )
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=1)

            if m.bias is not None:
                m.bias.data.fill_(self.bias)
