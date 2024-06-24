from utils import nn
from utils import *
import torch



class ReluExtractor(nn.Module):
    def __init__(self, model, device):
        super(ReluExtractor, self).__init__()
        self.model = model
        model.to(device)
        self.activations = []

        for name, layer in self.model.layers.named_modules():
            if isinstance(layer, nn.ReLU):
                layer.register_forward_hook(self.get_activation(name))

    def get_activation(self, name):
        def hook(module, input, output):
            self.activations.append(output.detach().cpu().numpy())
        return hook

    def forward(self, x):
        self.activations = []
        with torch.no_grad():
            output = self.model(x)
            return output, self.activations