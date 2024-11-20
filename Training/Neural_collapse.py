import torch
from torch.utils.data import DataLoader

import torch
from Data.DataLoader import get_simple_data_loader, get_data_loader
from Models.ForwardPass import *


# Compute NC1
def compute_nc1(all_features, all_labels, class_means, num_classes):
    within_class_variability = 0.0
    total_samples = 0

    for c in range(num_classes):
        class_features = all_features[all_labels == c]
        diffs = class_features - class_means[c]
        covariance = (diffs ** 2).sum()
        within_class_variability += covariance
        total_samples += class_features.numel()

    average_within_class_variability = within_class_variability / total_samples
    return average_within_class_variability.item()


# Compute NC2
def compute_nc2(class_means, num_classes):
    normalized_means = class_means / class_means.norm(dim=1, keepdim=True)
    cosine_similarities = torch.mm(normalized_means, normalized_means.T)
    off_diagonal_mask = ~torch.eye(num_classes, dtype=bool)
    off_diagonal_elements = cosine_similarities[off_diagonal_mask]
    mean_cosine_similarity = off_diagonal_elements.mean().item()
    expected_cosine_similarity = -1 / (num_classes - 1)
    return mean_cosine_similarity, expected_cosine_similarity


def compute_nc3(class_means, classifier_weights):
    normalized_means = class_means / class_means.norm(dim=1, keepdim=True)
    normalized_weights = classifier_weights / classifier_weights.norm(dim=1, keepdim=True)
    cos_similarities = (normalized_means * normalized_weights).sum(dim=1)
    mean_cos_similarity = cos_similarities.mean().item()
    return mean_cos_similarity


# Compute NC4
def compute_nc4(class_means, classifier_weights):
    mean_norms = class_means.norm(dim=1)
    weight_norms = classifier_weights.norm(dim=1)
    norm_ratios = mean_norms / weight_norms
    mean_norm_ratio = norm_ratios.mean().item()
    return mean_norm_ratio

def compute_neural_collapse(training_data, training_labels, model, feature_extractor, device, batch_size=1000):
    """
    Computes the Neural Collapse phenomena (NC1 to NC4) for the given model and dataset.

    Parameters:
    - training_data (torch.Tensor): Input data.
    - training_labels (torch.Tensor): Corresponding labels for the input data.
    - model (torch.nn.Module): Trained neural network model.
    - feature_extractor (torch.nn.Module): Part of the model used to extract features.
    - device (torch.device): Device to perform computations on ('cpu' or 'cuda').
    - batch_size (int): Batch size for processing data.

    Returns:
    - dict: A dictionary containing NC1 to NC4 metrics.
    """


    # Initialize lists to store features and labels
    all_features = []
    all_labels = []

    # # Create a DataLoader
    # dataset = TensorDataset(training_data, training_labels)
    # data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    data_loader = get_data_loader(training_data, training_labels, batch_size=batch_size, shuffle=False)

    # Set model to evaluation mode
    model.eval()

    with torch.no_grad():
        for sample, target in data_loader:
            sample = sample.to(device)
            target = target.to(device)

            # Extract features using the feature_extractor
            # Assuming hook_forward_train returns (output, activations)
            output, relu_outputs = hook_forward_train(feature_extractor, sample)
            # Assuming features are the outputs from the penultimate layer
            features = relu_outputs[-1]

            # all_features.append(features.cpu())
            # all_labels.append(target.cpu())
            all_features.append(features)
            all_labels.append(target)

    # Concatenate all features and labels
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)

    # Compute class means
    num_classes = len(torch.unique(all_labels))
    feature_dim = all_features.shape[1]
    class_means = torch.zeros((num_classes, feature_dim))

    for c in range(num_classes):
        class_features = all_features[all_labels == c]
        class_mean = class_features.mean(dim=0)
        class_means[c] = class_mean

    class_means = class_means.to(device)

    nc1_value = compute_nc1(all_features, all_labels, class_means, num_classes)

    mean_cos_sim, expected_cos_sim = compute_nc2(class_means, num_classes)

    # Compute NC3
    # Extract classifier weights (adjust this according to your model)
    classifier_weights = model.layers[-1].weight

    nc3_cos_sim = compute_nc3(class_means, classifier_weights)


    nc4_ratios = compute_nc4(class_means, classifier_weights)

    # Return results in a dictionary
    results = {
        'NC1': nc1_value,
        'NC2_mean_cosine_similarity': mean_cos_sim,
        # 'NC2_expected_cosine_similarity': expected_cos_sim
        'NC3': nc3_cos_sim,
        'NC4': nc4_ratios
    }


    # Center features within each class
    centered_features = []
    centered_labels = []
    for c in range(num_classes):
        class_features = all_features[all_labels == c]
        class_mean = class_means[c]
        centered = class_features - class_mean
        centered_features.append(centered)
        centered_labels.append(torch.full((class_features.shape[0],), c))
    
    # Concatenate centered features and labels
    centered_features = torch.cat(centered_features)
    centered_labels = torch.cat(centered_labels)
    
    # Compute within-class covariance matrices and perform eigenvalue decomposition
    covariance_matrices = {}
    eigenvalues = {}
    eigenvectors = {}
    manifold_dimensions = {}
    
    for c in range(num_classes):
        class_centered_features = centered_features[centered_labels == c]
        N_c = class_centered_features.shape[0]
        
        # Compute covariance matrix
        Sigma_c = (class_centered_features.T @ class_centered_features) / N_c  # Shape: (feature_dim, feature_dim)
        
        # Perform eigenvalue decomposition
        eigvals, eigvecs = torch.linalg.eigh(Sigma_c)
        # Sort eigenvalues and eigenvectors in descending order
        eigvals, eigvecs = eigvals.flip(0), eigvecs.flip(1)
        
        # Store results
        covariance_matrices[c] = Sigma_c
        eigenvalues[c] = eigvals
        eigenvectors[c] = eigvecs
        
        # Estimate manifold dimensionality
        explained_variance_ratio = eigvals / eigvals.sum()
        cumulative_evr = torch.cumsum(explained_variance_ratio, dim=0)
        
        # Choose dimensionality k_c where cumulative EVR reaches a threshold (e.g., 95%)
        threshold = 0.95
        k_c = torch.searchsorted(cumulative_evr, threshold).item() + 1  # +1 because indices start from 0
        manifold_dimensions[c] = k_c
    
    results2 = {
        # 'covariance_matrices': covariance_matrices,
        # 'eigenvalues': eigenvalues,
        # 'eigenvectors': eigenvectors,
        'total_variance': eigvals.sum().item(),
        'manifold_dimensions': manifold_dimensions,
        'mean manifold_dimenaion': np.mean(manifold_dimensions)
        # 'class_means': class_means,
        # 'centered_features': centered_features,
        # 'centered_labels': centered_labels
    }

    results.update(results2)

    return results
