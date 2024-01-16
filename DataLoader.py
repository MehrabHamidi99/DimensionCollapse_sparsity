import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from DataGenerator import *
        
def get_mnist_data_loaders(batch_size=256):
    # define transforms
    train, val, test = creat_mnist_data()

    train_loader =  DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader =  DataLoader(val, batch_size=1000, shuffle=True)
    test_loader =  DataLoader(test, batch_size=1000, shuffle=True)
    return train_loader, val_loader, test_loader


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