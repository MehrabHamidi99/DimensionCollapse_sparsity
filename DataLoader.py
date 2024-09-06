import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from DataGenerator import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

        
def get_mnist_data_loaders(batch_size=64):

    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST dataset
    ])

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Split the training dataset into training (50,000) and validation (10,000) sets
    train_size = 500
    val_size = 100
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))

    train_dataset_split = Subset(train_dataset, train_indices)
    val_dataset_split = Subset(train_dataset, val_indices)

    # Create DataLoaders for train, validation, and test datasets
    batch_size = 64  # You can change this according to your requirements

    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Faster extraction of samples and labels using direct indexing
    train_samples = train_dataset.data[train_indices].unsqueeze(1).float().view(-1, 28*28) / 255.0
    train_labels = train_dataset.targets[train_indices]

    val_samples = train_dataset.data[val_indices].unsqueeze(1).float().view(-1, 28*28) / 255.0
    val_labels = train_dataset.targets[val_indices]

    test_samples = test_dataset.data.unsqueeze(1).float().view(-1, 28*28) / 255.0
    test_labels = test_dataset.targets

    # Normalize the data (since we used .ToTensor() previously in the transformation)
    train_samples = (train_samples - 0.1307) / 0.3081
    val_samples = (val_samples - 0.1307) / 0.3081
    test_samples = (test_samples - 0.1307) / 0.3081

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels

def get_data_loader(X, y, batch_size=32):
    '''
    Parameters:
    X (numpy.ndarray): Input data.
    y (numpy.ndarray): Output data.
    batch_size (int, optional): Batch size for the DataLoader. Default is 32.
    
    Returns:
    DataLoader: A PyTorch DataLoader for the provided dataset.
    
    Description:
    get_data_loader converts the given numpy arrays X and y into PyTorch tensors and then creates a DataLoader. 
    This DataLoader can be used to iterate over the dataset in batches, suitable for training neural network models in PyTorch.

    Usage Example:
    loader = get_data_loader(X_train, y_train, batch_size=64)
'''
    tensor_x = torch.Tensor(X)
    tensor_y = torch.Tensor(y)

    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)