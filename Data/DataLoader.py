import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
from Data.DataGenerator import *
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from collections import Counter
from Data.Datasets import *

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


def get_mnist_data_loaders_odd_even(batch_size=32):
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

    train_dataset_split, val_dataset_split = torch.utils.data.random_split(train_dataset, [train_size, val_size]) # type: ignore

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


def get_mnist_data_loaders_three_class(batch_size=32, selected_classes=(0, 1, 2)):
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

    train_dataset_split, val_dataset_split = torch.utils.data.random_split(train_dataset, [train_size, val_size]) # type: ignore

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



def get_mnist_data_loaders_cnn_based_model(batch_size=64, image_size=28):

    # Define transformations for the data
    test_transform = transforms.Compose([transforms.Resize([image_size, image_size]), 
                                                transforms.ToTensor(), 
                                                transforms.Normalize([0.5], [0.5])])
    
    train_transform = transforms.Compose([transforms.Resize([image_size, image_size]),
                                            transforms.RandomCrop(image_size, padding=2), 
                                            transforms.ToTensor(), 
                                            transforms.Normalize([0.5], [0.5])])

    # Download MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

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
    train_samples = train_dataset.data[train_indices].unsqueeze(1).float() / 255.0
    train_labels = train_dataset.targets[train_indices]

    val_samples = train_dataset.data[val_indices].unsqueeze(1).float() / 255.0
    val_labels = train_dataset.targets[val_indices]

    test_samples = test_dataset.data.unsqueeze(1).float() / 255.0
    test_labels = test_dataset.targets

    # Normalize the data (since we used .ToTensor() previously in the transformation)
    train_samples = (train_samples - 0.1307) / 0.3081
    val_samples = (val_samples - 0.1307) / 0.3081
    test_samples = (test_samples - 0.1307) / 0.3081

    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels



def get_cifar100_data_loaders(batch_size=64):
    """
    Load CIFAR-100 dataset and create DataLoaders for training, validation, and testing.

    Args:
        batch_size (int, optional): Number of samples per batch. Defaults to 64.

    Returns:
        Tuple containing:
            - train_loader (DataLoader): DataLoader for training data.
            - val_loader (DataLoader): DataLoader for validation data.
            - test_loader (DataLoader): DataLoader for test data.
            - train_samples (torch.Tensor): Training samples tensor.
            - train_labels (torch.Tensor): Training labels tensor.
            - val_samples (torch.Tensor): Validation samples tensor.
            - val_labels (torch.Tensor): Validation labels tensor.
            - test_samples (torch.Tensor): Test samples tensor.
            - test_labels (torch.Tensor): Test labels tensor.
    """
    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-100 dataset
    ])

    # Download CIFAR-100 dataset
    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    # Split the training dataset into training (80%) and validation (20%) sets
    total_train_size = len(train_dataset)  # 50,000 for CIFAR-100
    train_size = int(0.8 * total_train_size)  # 40,000
    val_size = total_train_size - train_size  # 10,000

    # Create indices for splitting
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, total_train_size))

    # Create subset datasets
    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(train_dataset, val_indices)

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # Extract samples and labels
    # Note: Subset.dataset.data is a NumPy array for torchvision datasets
    train_samples = torch.tensor(train_dataset.data[train_indices], dtype=torch.float32) / 255.0
    val_samples = torch.tensor(train_dataset.data[val_indices], dtype=torch.float32) / 255.0
    test_samples = torch.tensor(test_dataset.data, dtype=torch.float32) / 255.0

    # Normalize the data (since we used transforms.Normalize with mean=0.5 and std=0.5)
    train_samples = (train_samples - 0.5) / 0.5
    val_samples = (val_samples - 0.5) / 0.5
    test_samples = (test_samples - 0.5) / 0.5

    # Convert labels to tensors
    train_labels = torch.tensor(train_dataset.targets, dtype=torch.long)[train_indices]
    val_labels = torch.tensor(train_dataset.targets, dtype=torch.long)[val_indices]
    test_labels = torch.tensor(test_dataset.targets, dtype=torch.long)

    # Optional: Print dataset statistics or status
    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, 
                 val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, \
            train_samples, train_labels, \
            val_samples, val_labels, \
            test_samples, test_labels


def get_fashion_mnist_data_loaders(batch_size=64):
    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))  # Normalize with mean and std of Fashion MNIST dataset
    ])

    # Download Fashion MNIST dataset
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    # Split the training dataset into training (50,000) and validation (10,000) sets
    train_size = 50000
    val_size = 10000
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))

    train_dataset_split = Subset(train_dataset, train_indices)
    val_dataset_split = Subset(train_dataset, val_indices)

    # Create DataLoaders for train, validation, and test datasets
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

    # Normalize the data
    train_samples = (train_samples - 0.2860) / 0.3530
    val_samples = (val_samples - 0.2860) / 0.3530
    test_samples = (test_samples - 0.2860) / 0.3530

    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels

def get_cifar10_data_loaders(batch_size=64):
    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10 dataset
    ])

    # Download CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Split the training dataset into training 0.8 and validation sets
    train_size = int(train_dataset.data.shape[0] * 0.8)
    val_size = train_dataset.data.shape[0] - int(train_dataset.data.shape[0] * 0.8)

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))

    train_dataset_split = Subset(train_dataset, train_indices)
    val_dataset_split = Subset(train_dataset, val_indices)

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract samples and labels
    train_samples, train_labels = train_dataset.data[train_indices], torch.tensor(train_dataset.targets)[train_indices]
    val_samples, val_labels = train_dataset.data[val_indices], torch.tensor(train_dataset.targets)[val_indices]
    test_samples, test_labels = test_dataset.data, torch.tensor(test_dataset.targets)

    # Normalize the data (since we used .ToTensor() previously in the transformation)
    train_samples = (train_samples.astype('float32') / 255.0 - 0.5) / 0.5
    val_samples = (val_samples.astype('float32') / 255.0 - 0.5) / 0.5
    test_samples = (test_samples.astype('float32') / 255.0 - 0.5) / 0.5

    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels


def get_cifar10_data_loaders_resnetMLP(batch_size=64):
    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10 dataset
    ])

    # Download CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Create custom datasets with reshaped images
    train_dataset = CustomCIFAR10Dataset(train_dataset)
    test_dataset = CustomCIFAR10Dataset(test_dataset)

    # Split the training dataset into training 0.8 and validation sets
    train_size = int(train_dataset.data.shape[0] * 0.8)
    val_size = train_dataset.data.shape[0] - int(train_dataset.data.shape[0] * 0.8)

    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))

    train_dataset_split = Subset(train_dataset, train_indices)
    val_dataset_split = Subset(train_dataset, val_indices)

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract samples and labels
    train_samples, train_labels = train_dataset.data[train_indices], torch.tensor(train_dataset.targets)[train_indices]
    val_samples, val_labels = train_dataset.data[val_indices], torch.tensor(train_dataset.targets)[val_indices]
    test_samples, test_labels = test_dataset.data, torch.tensor(test_dataset.targets)

    # Normalize the data (since we used .ToTensor() previously in the transformation)
    train_samples = (train_samples / 255.0 - 0.5) / 0.5
    val_samples = (val_samples / 255.0 - 0.5) / 0.5
    test_samples = (test_samples / 255.0 - 0.5) / 0.5

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


def get_simple_data_loader(X, batch_size=10000, shuffle=False):
    
    tensor_x = torch.Tensor(X)

    dataset = MyDataset(tensor_x)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __getitem__(self, index):
        x = self.data[index]
        return x
    
    def __len__(self):
        return len(self.data)
    


def get_kuzushiji_mnist_data_loaders(batch_size=64):
    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1904,), (0.3475,))  # Normalize with mean and std of Kuzushiji-MNIST dataset
    ])

    # Download Kuzushiji-MNIST dataset
    train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

    # Split the training dataset into training (50,000) and validation (10,000) sets
    train_size = 50000
    val_size = 10000
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))

    train_dataset_split = Subset(train_dataset, train_indices)
    val_dataset_split = Subset(train_dataset, val_indices)

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Faster extraction of samples and labels using direct indexing
    train_samples = train_dataset.data[train_indices].unsqueeze(1).float().view(-1, 28 * 28) / 255.0
    train_labels = train_dataset.targets[train_indices]

    val_samples = train_dataset.data[val_indices].unsqueeze(1).float().view(-1, 28 * 28) / 255.0
    val_labels = train_dataset.targets[val_indices]

    test_samples = test_dataset.data.unsqueeze(1).float().view(-1, 28 * 28) / 255.0
    test_labels = test_dataset.targets

    # Normalize the data (since we used .ToTensor() previously in the transformation)
    train_samples = (train_samples - 0.1904) / 0.3475
    val_samples = (val_samples - 0.1904) / 0.3475
    test_samples = (test_samples - 0.1904) / 0.3475

    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels



def get_three_class_cifar10_data_loaders(batch_size=64, selected_classes=[0, 1, 2]):
    # Define transformations for the data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize for CIFAR-10 dataset
    ])

    # Download CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Filter the dataset for the selected classes
    def filter_classes(dataset, selected_classes):
        indices = [i for i, label in enumerate(dataset.targets) if label in selected_classes]
        dataset.targets = torch.tensor(dataset.targets)[indices]
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

    # Split the training dataset into training (80%) and validation (20%) sets
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, train_size + val_size))

    train_dataset_split = Subset(train_dataset, train_indices)
    val_dataset_split = Subset(train_dataset, val_indices)

    # Create DataLoaders for train, validation, and test datasets
    train_loader = DataLoader(train_dataset_split, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset_split, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Extract samples and labels
    train_samples, train_labels = train_dataset.data[train_indices], train_dataset.targets[train_indices]
    val_samples, val_labels = train_dataset.data[val_indices], train_dataset.targets[val_indices]
    test_samples, test_labels = test_dataset.data, test_dataset.targets

    # Normalize the data (since we used .ToTensor() previously in the transformation)
    train_samples = (train_samples.astype('float32') / 255.0 - 0.5) / 0.5
    val_samples = (val_samples.astype('float32') / 255.0 - 0.5) / 0.5
    test_samples = (test_samples.astype('float32') / 255.0 - 0.5) / 0.5

    print_status(train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels)

    return train_loader, val_loader, test_loader, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels

