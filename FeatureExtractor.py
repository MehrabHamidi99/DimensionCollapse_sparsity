from utils import nn
from utils import *
import torch



class ReluExtractor(nn.Module):
    def __init__(self, model, device):
        super(ReluExtractor, self).__init__()
        self.model = model.to(device)
        self.activations = []

        for name, module in model.named_modules():
            # if isinstance(module, nn.ReLU):
            if isinstance(module, nn.Linear) or isinstance(module, nn.LogSoftmax):
                module.register_forward_hook(self.get_activation())

    def get_activation(self):
        def hook(module, input, output):
            # Store the activations (output of ReLU layer)
            self.activations.append(output.cpu())  # Use .cpu() to make it easier to work with activations on any device
        return hook

    def forward(self, x):
        self.activations = []
        with torch.no_grad():
            output = self.model(x)
        return output, self.activations