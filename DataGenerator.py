import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader



def gaussian_hypersphere(D, N=1000, r=1, surface=True):
    """
    Generates points uniformly distributed inside (or on the surface of) a D-dimensional hypersphere.

    Parameters:
    N (int): Number of points to generate.
    D (int): Dimension of the hypersphere.
    r (float, optional): Radius of the hypersphere. Default is 1.
    surface (bool, optional): If True, points will be on the surface of the hypersphere. Default is False.

    Returns:
    numpy.ndarray: An array of shape (N, D) representing the points.
    """

    # Set seed for reproducibility
    np.random.seed(1)

    # Sample D vectors of N Gaussian coordinates
    samples = np.random.randn(N, D)

    # Normalize all distances (radii) to 1
    radii = np.linalg.norm(samples, axis=1)
    samples = samples / radii[:, np.newaxis]

    # Sample N radii with exponential distribution (unless points are to be on the surface)
    if not surface:
        new_radii = np.random.uniform(size=N) ** (1 / D)
        samples = samples * new_radii[:, np.newaxis]

    return samples * r


def create_random_data_uniform(input_dimension, num=1000):
    # Generate random input data
    return np.random.rand(num, input_dimension)

def create_random_data_normal_dist(input_dimension, num=1000, loc=0, scale=1):
    # Generate random input data
    return np.random.normal(loc=loc, scale=scale, size=(num, input_dimension))

def create_random_data(input_dimension, num=1000, normal_dsit=False, loc=0, scale=1, exp_type='normal', constant=5):
    '''
    Parameters:
    input_dimension (int): The number of features for each input sample.
    num (int, optional): The total number of data points to generate. Default is 1000.

    Returns:
    Tuple of (X_normalized, y_normalized), where:
    X_normalized (numpy.ndarray): The normalized input features.
    y_normalized (numpy.ndarray): The normalized output values.

    Description:
    create_random_data generates a dataset of num samples, each with input_dimension features. 
    The output values are linear combinations of the input features with added Gaussian noise. 
    Both input and output data are normalized.

    Usage Example:
    X, y = create_random_data(5, 1000)

    '''
    if exp_type == 'normal':
        if normal_dsit:
            X = create_random_data_normal_dist(input_dimension, num, loc, scale)
        
        else:
            X = create_random_data_uniform(input_dimension, num)

    elif exp_type == 'fixed':
        X = gaussian_hypersphere(input_dimension, num, r=constant)


    # Define a simple linear relationship (for simplicity, using a vector of ones as coefficients)
    coefficients = np.random.rand(1, input_dimension).flatten()

    # Calculate output data with a linear transformation
    y = np.dot(X, coefficients)
    # Add some noise to y
    noise = np.random.normal(0, 0.1, num)  # Gaussian noise with mean 0 and standard deviation 0.1
    y += noise

    # Normalize the dataset
    # X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    y_normalized = (y - np.mean(y)) / np.std(y)

    return X, y_normalized


def create_full_random_data(input_dimension, output_dim=1, train_num=800, val_num=500, test_num=100, normal_dsit=False, loc=0, scale=1):
    '''
    Parameters:
    input_dimension (int): The number of features for each input sample.
    train_num (int, optional): Number of training samples. Default is 800.
    val_num (int, optional): Number of validation samples. Default is 500.
    test_num (int, optional): Number of testing samples. Default is 100.
    
    Returns:
    Tuple of three tuples: (train_data, val_data, test_data), 
    where each tuple contains normalized input and output data (X_normalized, y_normalized).
    
    Description:
    create_full_random_data creates datasets for training, validation, and testing. 
    It generates a total of train_num + val_num + test_num samples and splits them into the three datasets. 
    Each sample comprises input_dimension features, and the output is a linear combination of these features with added Gaussian noise.

    Usage Example:
    train_data, val_data, test_data = create_full_random_data(10, 800, 200, 100)
    '''
    total_num = train_num + val_num + test_num

    if normal_dsit:
        X = create_random_data_normal_dist(input_dimension, total_num, loc, scale)
    
    else:
        X = create_random_data_uniform(input_dimension, total_num)

    # # Generate random input data
    # X = np.random.rand(total_num, input_dimension)

    # Define a simple linear relationship
    coefficients = np.random.normal(0, 1, (input_dimension, output_dim))

    # Calculate output data with a linear transformation
    y = np.dot(X, coefficients)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.1, (total_num, output_dim))
    y += noise

    # Normalize the dataset
    # X_normalized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    X_normalized = X
    # y_normalized = (y - np.mean(y)) / np.std(y)
    y_normalized = y

    # Split into training, validation, and test sets
    return (X_normalized[:train_num], y_normalized[:train_num]), \
           (X_normalized[train_num:train_num + val_num], y_normalized[train_num:train_num + val_num]), \
           (X_normalized[-test_num:], y_normalized[-test_num:])


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