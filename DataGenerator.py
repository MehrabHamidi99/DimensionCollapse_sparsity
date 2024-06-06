from utils import *


def gaussian_hypersphere(D, N=1000, r=1, surface=True):
    """
    Generates points uniformly distributed inside (or on the surface of) a D-dimensional hypersphere.

    Parameters:
    N (int): Number of points to generate.
    D (int): Dimension of the hypersphere.
    r (float, optional): Radius of the hypersphere. Default is 1.
    surface (bool, optional): If True, points will be on the surface of the hypersphere. Default is True.

    Returns:
    numpy.ndarray: An array of shape (N, D) representing the points.
    """

    # Sample D vectors of N Gaussian coordinates
    samples = np.random.randn(N, D)

    # Normalize all distances (radii) to 1
    # radii = np.linalg.norm(samples, axis=1)
    radii = np.sqrt(np.einsum('...i,...i', samples, samples))
    samples = samples / radii[:, np.newaxis]

    # Sample N radii with exponential distribution (unless points are to be on the surface)
    if not surface:
        new_radii = np.random.uniform(size=N) ** (1 / D)
        samples = samples * new_radii[:, np.newaxis]

    return samples * r

def create_2d_plane_samples(d, N=1000, loc=0, scale=1):
    """
    Sample points on a 2D plane in a D-dimensional space.

    Parameters:
    - d (int): Dimensionality of the space.
    - n_samples (int): Number of points to sample on the plane.
    - origin (np.array): A point on the plane (D-dimensional).
    - basis_vector_1 (np.array): First basis vector defining the plane (D-dimensional).
    - basis_vector_2 (np.array): Second basis vector defining the plane (D-dimensional).

    Returns:
    - np.array: Array of points sampled from the plane (n_samples x d).
    """
    origin = np.zeros(d)  # Origin point on the plane
    basis_vector_1 = np.random.rand(d)  # First basis vector
    basis_vector_1 /= np.linalg.norm(basis_vector_1)  # Normalize
    basis_vector_2 = np.random.rand(d)  # Second basis vector orthogonal to the first
    basis_vector_2 -= basis_vector_2.dot(basis_vector_1) * basis_vector_1  # Make orthogonal
    basis_vector_2 /= np.linalg.norm(basis_vector_2)  # Normalize
    
    # Sample scalar coefficients for the linear combinations
    coefficients_1 = np.random.normal(loc, scale, N)
    coefficients_2 = np.random.normal(loc, scale, N)

    # Calculate the points
    points = origin + np.outer(coefficients_1, basis_vector_1) + np.outer(coefficients_2, basis_vector_2)
    
    return points

def points_on_a_line(D, N=1000, loc=0, scale=1, d2d=True):
    samples = np.random.normal(loc=loc, scale=scale, size=(N, D))
    m = np.random.rand(1, D - 1)
    b = np.random.rand(1, 1)
    if d2d:
        m = np.random.rand(1, 1)
        b = np.random.rand(1, 1)
        samples[:, 1] = (np.dot(samples[:, 0][:, None], m.T) + b).flatten()
        samples[:, 2:] = 0
    else:
        samples[:, -1] = (np.dot(samples[:, :-1], m.T) + b).flatten()

    return samples



def create_random_data_uniform(input_dimension, num=1000):
    # Generate random input data
    return np.random.rand(num, input_dimension)

def create_random_data_normal_dist(input_dimension, num=1000, loc=0, scale=1):
    # Generate random input data
    return np.random.normal(loc=loc, scale=scale, size=(num, input_dimension))

def create_random_data(input_dimension, num=1000, normal_dsit=True, loc=0, scale=1, exp_type='normal', constant=5):
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
        X = gaussian_hypersphere(input_dimension, num, r=scale)
        
    elif exp_type == 'line':
        X = points_on_a_line(input_dimension, num, scale=scale)

    elif exp_type == 'line_d':
        X = points_on_a_line(input_dimension, num, scale=scale, d2d=False)
    
    elif exp_type == 'plane':
        X = create_2d_plane_samples(input_dimension, num, loc=loc, scale=scale)
    else:
        print(exp_type)
        raise Exception("Unknonw type data")


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

def creat_mnist_data():
    transform = transforms.Compose([transforms.ToTensor()
                                    # , transforms.Normalize((0.5,), (0.5,))
                                    ])
    dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    return train_set, val_set, test_dataset