import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from DataGenerator import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter


def print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels):
    # Print dataloader batch shapes
    print("Train Loader Batch Shapes:")
    for i, (images, labels) in enumerate(train_loader):
        print(f"Batch {i+1}: Images Shape = {images.shape}, Labels Shape = {labels.shape}")
        break  # We only print the shape of the first batch to avoid flooding

    print("\nValidation Loader Batch Shapes:")
    for i, (images, labels) in enumerate(val_loader):
        print(f"Batch {i+1}: Images Shape = {images.shape}, Labels Shape = {labels.shape}")
        break

    print("\nTest Loader Batch Shapes:")
    for i, (images, labels) in enumerate(test_loader):
        print(f"Batch {i+1}: Images Shape = {images.shape}, Labels Shape = {labels.shape}")
        break

    # Print dataset shapes
    print("\nTrain Samples Shape:", train_samples.shape)
    print("Train Labels Shape:", train_labels.shape)

    print("\nValidation Samples Shape:", val_samples.shape)
    print("Validation Labels Shape:", val_labels.shape)

    print("\nTest Samples Shape:", test_samples.shape)
    print("Test Labels Shape:", test_labels.shape)

    # Compute label frequencies
    train_label_freq = Counter(train_labels.numpy())
    val_label_freq = Counter(val_labels.numpy())
    test_label_freq = Counter(test_labels.numpy())

    # Print label frequencies
    print("\nTrain Label Frequencies:", train_label_freq)
    print("Validation Label Frequencies:", val_label_freq)
    print("Test Label Frequencies:", test_label_freq)


def get_mnist_data_loaders_odd_even(batch_size=64):
    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST dataset
    ])

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter the dataset for odd and even classes
    even_classes = [0, 2, 4, 6, 8]
    odd_classes = [1, 3, 5, 7, 9]

    def filter_odd_even(dataset):
        indices = [i for i, label in enumerate(dataset.targets) if label in even_classes or label in odd_classes]
        dataset.targets = dataset.targets[indices]
        dataset.data = dataset.data[indices]
        return dataset

    # Apply the filtering
    train_dataset = filter_odd_even(train_dataset)
    test_dataset = filter_odd_even(test_dataset)

    # Map the selected classes to new labels (0: Even, 1: Odd)
    def remap_odd_even_labels(dataset):
        dataset.targets = torch.tensor([0 if label.item() in even_classes else 1 for label in dataset.targets])

    remap_odd_even_labels(train_dataset)
    remap_odd_even_labels(test_dataset)

    # Split the training dataset into training (filtered size) and validation (17% for validation)
    train_size = int(0.83 * len(train_dataset))  # 83% for training
    val_size = len(train_dataset) - train_size  # 17% for validation

    train_dataset_split, val_dataset_split = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Faster extraction of samples and labels using direct indexing
    train_samples = train_dataset.data[train_dataset_split.indices].unsqueeze(1).float().view(-1, 28*28) / 255.0
    train_labels = train_dataset.targets[train_dataset_split.indices]

    val_samples = train_dataset.data[val_dataset_split.indices].unsqueeze(1).float().view(-1, 28*28) / 255.0
    val_labels = train_dataset.targets[val_dataset_split.indices]

    test_samples = test_dataset.data.unsqueeze(1).float().view(-1, 28*28) / 255.0
    test_labels = test_dataset.targets

    # Normalize the data (since we used .ToTensor() previously in the transformation)
    train_samples = (train_samples - 0.1307) / 0.3081
    val_samples = (val_samples - 0.1307) / 0.3081
    test_samples = (test_samples - 0.1307) / 0.3081

    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels


def get_mnist_data_loaders_three_class(batch_size=64, selected_classes=(0, 1, 2)):
    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Normalize with mean and std of MNIST dataset
    ])

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Filter the dataset for the selected classes
    def filter_classes(dataset, selected_classes):
        indices = [i for i, label in enumerate(dataset.targets) if label in selected_classes]
        dataset.targets = dataset.targets[indices]
        dataset.data = dataset.data[indices]
        return dataset

    # Apply the filtering
    train_dataset = filter_classes(train_dataset, selected_classes)
    test_dataset = filter_classes(test_dataset, selected_classes)

    # Map the selected classes to new labels (0, 1, 2)
    def remap_labels(dataset, selected_classes):
        class_map = {c: i for i, c in enumerate(selected_classes)}
        dataset.targets = torch.tensor([class_map[label.item()] for label in dataset.targets])

    remap_labels(train_dataset, selected_classes)
    remap_labels(test_dataset, selected_classes)

    # Split the training dataset into training (filtered size) and validation (10,000) sets
    train_size = int(0.83 * len(train_dataset))  # 83% for training
    val_size = len(train_dataset) - train_size  # 17% for validation

    train_dataset_split, val_dataset_split = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Faster extraction of samples and labels using direct indexing
    train_samples = train_dataset.data[train_dataset_split.indices].unsqueeze(1).float().view(-1, 28*28) / 255.0
    train_labels = train_dataset.targets[train_dataset_split.indices]

    val_samples = train_dataset.data[val_dataset_split.indices].unsqueeze(1).float().view(-1, 28*28) / 255.0
    val_labels = train_dataset.targets[val_dataset_split.indices]

    test_samples = test_dataset.data.unsqueeze(1).float().view(-1, 28*28) / 255.0
    test_labels = test_dataset.targets

    # Normalize the data (since we used .ToTensor() previously in the transformation)
    train_samples = (train_samples - 0.1307) / 0.3081
    val_samples = (val_samples - 0.1307) / 0.3081
    test_samples = (test_samples - 0.1307) / 0.3081

    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels

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
    train_size = 50000
    val_size = 10000
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

    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels

def get_data_loader(X, y, batch_size=32, shuffle=True):
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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)