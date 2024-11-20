from utils import nn
from utils import *
import torch
from torch.utils.hooks import RemovableHandle
import contextlib

@contextlib.contextmanager
def collect_activations(module: nn.Module):

    activations: list[torch.Tensor] = []
    handles: list[RemovableHandle] = []
    names: list[str] = []

    def _save_activation_hook(module, input, output) -> None:
        assert isinstance(output, torch.Tensor)
        activations.append(output)

    for name, module in module.named_modules():
        # if isinstance(module, nn.ReLU) or isinstance(module, nn.LogSoftmax):
        # if isinstance(module, nn.Linear):
        if isinstance(module, nn.ReLU):
            handle = module.register_forward_hook(_save_activation_hook)
            handles.append(handle)
            names.append(name)
    # Yield, during which the model does a forward pass
    yield activations, names

    for handle in handles:
        handle.remove()




class ReluExtractor(nn.Module):
    def __init__(self, model, device):
        super(ReluExtractor, self).__init__()
        self.model = model.to(device)
    def forward(self, x, names=False):
        with collect_activations(self.model) as activations:
            with torch.no_grad():
                output = self.model(x)
        if names:
            return output, activations[0], activations[1]
        return output, activations[0]