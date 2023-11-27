import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from Models import *
from utils import *
from DataGenerator import *

def stable_neuron_analysis(model, dataset, y=None):
    '''
    Parameters
    model (torch.nn.Module): A trained PyTorch neural network model.
    dataset (iterable): A dataset where each element is an input sample to the model.
    y (iterable, optional): Output labels for the dataset, not used in the current implementation.
    
    Functionality
    Iterates over each input sample in the dataset.
    Performs a forward pass through the model to obtain pre-activation and activation values for each layer.
    Invokes analysis_neurons_activations_depth_wise on the model to analyze neuron activations.
    
    Usage
    Primarily used for understanding which neurons are consistently active or inactive across a dataset.
    '''
    # Process each input in the dataset
    for x, y in dataset:
        # Get pre-activation and activation values for each layer
        _ = model(x, return_pre_activations=True)
        model.analysis_neurons_activations_depth_wise(len(dataset.dataset))


def train_model(model, train_loader, val_loader=None, epochs=50, learning_rate=0.001):
    '''
    Parameters
    model (torch.nn.Module): A PyTorch neural network model to be trained.
    train_loader (DataLoader): DataLoader containing the training dataset.
    val_loader (DataLoader, optional): DataLoader containing the validation dataset. If None, validation is skipped.
    epochs (int, optional): The number of epochs for which the model will be trained. Default is 50.
    learning_rate (float, optional): Learning rate for the optimizer. Default is 0.001.
    
    Returns
    val_analysis (list): A list containing the aggregated neuron activations for each epoch during validation.
    val_analysis_depth (list): A list containing the aggregated neuron activation ratios for each layer 
                                of the model for each epoch during validation.
    
    Functionality
    Trains the model using the Mean Squared Error loss and Adam optimizer.
    After each training epoch, evaluates the model on the validation dataset if provided.
    For each epoch during validation, performs an analysis of neuron activations using 
    the model's analysis_neurons_activations_depth_wise method.
    Collects and returns the neuron activation data across all epochs.
    
    Usage Example
    model = MLP_ReLU(n_in=10, layer_list=[20, 30, 1])
    train_loader = DataLoader(train_dataset, batch_size=32)
    val_loader = DataLoader(val_dataset, batch_size=32)
    val_analysis, val_analysis_depth = train_model(model, train_loader, val_loader)
    
    Notes
    This function is particularly useful for understanding how the activations of neurons in 
    a neural network evolve over the course of training.
    The neuron activation analysis can provide insights into which neurons are 
    consistently active or inactive and how this pattern changes with each epoch.
    The function assumes that the model has a method analysis_neurons_activations_depth_wise 
    to perform the activation analysis, which should be implemented in the model class.
    This function is a comprehensive tool for training neural network models while simultaneously analyzing their behavior, 
    particularly useful for research and in-depth study of neural network dynamics.
    '''
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    val_analysis = []

    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = F.mse_loss(outputs, y_batch)
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for x_val, y_val in val_loader:
                    val_outputs = model(x_val, return_pre_activations=False)
                    val_loss += F.mse_loss(val_outputs, y_val).item()
                    # model.analysis_neurons_activations_depth_wise()
                # print(f'Epoch {epoch+1}, Validation Loss: {val_loss / len(val_loader)}')
        
            stable_neuron_analysis(model, val_loader)
            val_analysis += [model.additive_activations]
            model.reset_all()

    return val_analysis