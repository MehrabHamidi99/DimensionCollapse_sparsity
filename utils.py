import pandas as pd
import numpy as np
np.random.seed(33)
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

from collections import defaultdict

import torch
torch.manual_seed(33)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.linear_model import LinearRegression, RANSACRegressor, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import seaborn as sns

import os
# os.chdir('/content/gdrive/MyDrive/DeepReluSymmetries/')
import seaborn as sns
import pandas as pd
import random
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms

from tqdm import tqdm
from scipy.spatial import ConvexHull
from scipy import spatial

import itertools

import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.patches as mpatches

import os
from PIL import Image, ImageDraw, ImageFont

def c(m):
    xy = np.dot(m, m.T) # O(k^3)
    x2 = y2 = (m * m).sum(1) #O(k^2)
    d2 = np.add.outer(x2, y2)- 2 * xy  #O(k^2)
    d2.flat[::len(m)+1]=0 # Rounding issues
    return d2
    # return np.sqrt(d2)  # O (k^2)

def distance_from_origin(vect_org):
    if not torch.is_tensor(vect_org):
        vect = torch.Tensor(vect_org)
    else:
        vect = vect_org.clone()
    # if torch.is_tensor(vect):
    #     # vect_cop = vect.cpu().clone().detach().numpy()
    #     return torch.sqrt(torch.sum(((vect_cop - torch.zeros(vect_cop.shape)) ** 2), axis=1))
    return torch.sqrt(torch.sum(((vect - torch.zeros(vect.shape, dtype=vect.dtype, device=vect.device)) ** 2), dim=1))

def get_pc_components(data, n_components=2):
    data = data - np.mean(data)
    covar_matrix = np.matmul(data.T , data)
    # Perform PCA
    d = covar_matrix.shape[0]
    if n_components == 3:
        values, vectors = eigh(covar_matrix, eigvals=(d - 3, d - 1), eigvals_only=False)
        projected_data = np.dot(data, vectors)
        if projected_data.shape[1] < 2:
            return [np.zeros(data[:, 0].shape), np.zeros(data[:, 0].shape), np.zeros(data[:, 0].shape)]
        return [projected_data[:, -1], projected_data[:, -2], projected_data[:, -3]]
    values, vectors = eigh(covar_matrix, eigvals=(d - 2, d - 1), eigvals_only=False)
    projected_data = np.dot(data, vectors)
    if projected_data.shape[1] < 2:
        return [np.zeros(data[:, 0].shape), np.zeros(data[:, 0].shape)]
    return [projected_data[:, -1], projected_data[:, -2]]

def count_near_zero_eigenvalues(data, threshold=1e-7, return_eigenvectors=False, partial=False):
    """
    Counts the number of eigenvalues of the covariance matrix of the data that are close to zero.

    Parameters:
    data (numpy.ndarray): A 2D numpy array where rows represent samples and columns represent features.
    threshold (float): A threshold to consider an eigenvalue as 'close to zero'. Default is 0.01.

    Returns:
    int: The number of eigenvalues close to zero.
    """
    
    normalized_data = data - np.mean(data)
    normalized_data = np.array(normalized_data, dtype=np.float64)

    # if partial:
    #     transformer = IncrementalPCA(n_components=7, batch_size=200)
    covar_matrix = np.dot(normalized_data.transpose() , normalized_data)
    if return_eigenvectors:
        values, vectors = eigh(covar_matrix, eigvals_only=False)

        near_zero_count = np.sum(np.abs(values) > threshold)
        return near_zero_count, values, vectors
    
    values = eigh(covar_matrix, b=np.eye(len(covar_matrix), dtype=covar_matrix.dtype), eigvals_only=True, turbo=True, check_finite=False)
    values[values < 0] = 0
    near_zero_count = np.sum(values > threshold)
    return near_zero_count, values


def projection_analysis(data, type_anal, dim):
    if data.shape[1] == 2:
        if dim == 3:
            return [data[:, 0], data[:, 1], np.zeros(data.shape[0])]
        return [data[:, 0], data[:, 1]]
    if data.shape[1] < 2:
        return [data[:, 0], data[:, 0]]
    if dim == 2:
        if type_anal == 'pca':
            return get_pc_components(data)
        elif type_anal == 'random':
            random_dims  = random.sample(set(list(range(0, data.shape[1]))), dim)
            return [data[0:data.shape[0], random_dims[0]], data[0:data.shape[0], random_dims[1]]]
    elif dim == 3:
        if type_anal == 'pca':
            return get_pc_components(data, n_components=3)
        elif type_anal == 'random':
            random_dims  = random.sample(set(list(range(0, data.shape[1]))), dim)
            return [data[0:data.shape[0], random_dims[0]],
                     data[0:data.shape[0], random_dims[1]],
                       data[0:data.shape[0], random_dims[2]]]
    raise Exception("type or dim set wrong!")

def file_name_handling(which, architecture, num='', exps=1, pre_path='', normal_dist=False, loc=0, scale=1, bias=1e-4, exp_type='normal', model_type='mlp', return_pre_path=False, new_model_each_time=False):
    if normal_dist:
        pre_path += '{}_mean_{}_std{}/'.format(str(exp_type), str(loc), str(scale))
    else:
        pre_path += 'uniform/'
    pre_path += 'bias_{}/'.format(str(bias))
    if return_pre_path:
        return pre_path
    pre_path = pre_path + which + 'bias{}_exps{}_num{}_center_{}'.format(str(bias), str(exps), str(num), str(loc))
    if new_model_each_time:
        pre_path += '_new_model_each_time'
    pre_path = pre_path + '_{}'.format(model_type)
    this_path = pre_path + '_{}/'.format(str(architecture))
    if not os.path.isdir(this_path):
        try:
            os.makedirs(this_path)
        except OSError as exc:
            this_path = pre_path + '_{}_{}_{}/'.format(str(architecture[0]), str(architecture[1][0]), str(len(architecture[1])))
            if not os.path.isdir(this_path):
                os.makedirs(this_path)
    print(this_path)
    return this_path

def calc_spherical_mean_width_v2(this_data_org, num_directions=1e3):
    """
    Calculate the spherical mean width of the given data using random directions.

    Args:
        this_data (torch.Tensor): Input data of shape (N, D), where N is the number of samples
                                  and D is the dimensionality.
        num_directions (int, optional): Number of random directions to sample. Defaults to 1000.

    Returns:
        torch.Tensor: The mean spherical width.
    """
    if not torch.is_tensor(this_data_org):
        this_data = torch.Tensor(this_data_org)
    else:
        this_data = this_data_org.clone()
    # Ensure num_directions is an integer
    num_directions = int(num_directions)
    
    # Generate random directions uniformly in [0, 1) and convert to double precision
    directions = torch.rand(num_directions, this_data.shape[1], dtype=this_data.dtype, device=this_data.device)  # Shape: (T, D)
    # Normalize each direction to lie on the unit sphere
    directions = directions / directions.norm(dim=1, keepdim=True)  # Shape: (T, D)
    # Compute the projection of data onto each direction
    # this_data: (N, D), directions.t(): (D, T) -> mean_width: (N, T)
    mean_width = torch.matmul(this_data, directions.t())  # Shape: (N, T)
    # Calculate the width for each direction by finding the range of projections
    width = torch.max(mean_width, dim=0).values - torch.min(mean_width, dim=0).values  # Shape: (T,)
    # Compute the mean width across all directions
    return width.mean()


def covariance_matrix_additional_and_projectional(covariance_matrix, result_dict, device, threshold=1e-7):
    # covar_matrix = covariance_matrix.cpu().detach().numpy()
    cov_mat = covariance_matrix.to(device)
    cov_mat += torch.eye(cov_mat.shape[0]).to(device) * 1e-6  # Add a small regularization term to the diagonal
    values, vectors = torch.linalg.eigh(cov_mat)
    values[values < 0] = 0

    near_zero_count = torch.sum(torch.abs(values) > threshold)

    result_dict['nonzero_eigenvalues_count'].append(near_zero_count.cpu().detach().numpy())
    result_dict['eigenvalues'].append(values.cpu().detach().numpy())

    singular_values = torch.sqrt(values)

    result_dict['stable_rank'].append((torch.sum(singular_values) / torch.max(singular_values)).cpu().detach().numpy())
    result_dict['simple_spherical_mean_width'].append((2 * torch.mean(singular_values)).cpu().detach().numpy())

    result_dict['eigenvectors'].append(vectors.cpu().detach().numpy())

    return result_dict



def additional_analysis_for_full_data(this_data, result_dict):
        
    tmp = count_near_zero_eigenvalues(this_data, return_eigenvectors=False)
    result_dict['nonzero_eigenvalues_count'].append(tmp[0])
    result_dict['eigenvalues'].append(tmp[1])
    singular_values = np.sqrt(tmp[1])
    result_dict['stable_rank'].append(np.sum(singular_values) / (np.max(singular_values) + 1e-7))
    result_dict['simple_spherical_mean_width'].append(2 * np.mean(singular_values))
    result_dict['spherical_mean_width_v2'].append(calc_spherical_mean_width_v2(this_data).detach().cpu().numpy().tolist())
    
    # cdist(this_data, np.array([np.zeros(this_data.shape[1])]))
    result_dict['norms'].append(distance_from_origin(this_data).detach().cpu().numpy().tolist())
    # [np.mean(distances_this_data), np.max(distances_this_data), np.min(distances_this_data)]

    return result_dict

def projection_analysis_for_full_data(this_data, results, return_eigenvalues=True):
    res_list = count_near_zero_eigenvalues(this_data, return_eigenvectors=return_eigenvalues)

    results['eigenvectors'].append(res_list[2])
    results['pca_2'].append(projection_analysis(this_data, 'pca', 2))
    results['pca_3'].append(projection_analysis(this_data, 'pca', 3))
    results['random_2'].append(projection_analysis(this_data, 'random', 2))
    results['random_3'].append(projection_analysis(this_data, 'random', 3))

    return results

def perform_pca_and_analyses(results_dict, device):

    results_dict['pca_projections_2d'] = []
    results_dict['pca_projections_3d'] = []

    for i in range(len(results_dict['layer_activations'])):
        activations = results_dict['layer_activations'][i]
        # Perform PCA for 2D projection
        pca_2d = PCA(n_components=2)
        projections_2d = pca_2d.fit_transform(activations)
        results_dict['pca_projections_2d'].append(projections_2d)

        # Perform PCA for 3D projection
        pca_3d = PCA(n_components=3)
        projections_3d = pca_3d.fit_transform(activations)
        results_dict['pca_projections_3d'].append(projections_3d)

    return results_dict