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

from utils_plotting import *

def c(m):
    xy = np.dot(m, m.T) # O(k^3)
    x2 = y2 = (m * m).sum(1) #O(k^2)
    d2 = np.add.outer(x2, y2)- 2 * xy  #O(k^2)
    d2.flat[::len(m)+1]=0 # Rounding issues
    return d2
    # return np.sqrt(d2)  # O (k^2)

def distance_from_origin(vect):
    if torch.is_tensor(vect):
        vect_cop = vect.cpu().clone().detach().numpy()
        return np.sqrt(np.sum(((vect_cop - np.zeros(vect_cop.shape)) ** 2), axis=1))
    return np.sqrt(np.sum(((vect - np.zeros(vect.shape)) ** 2), axis=1))

def get_pc_components(data, n_components=2):
    data = data - np.mean(data)
    covar_matrix = np.matmul(data.T , data)
    # Perform PCA
    d = covar_matrix.shape[0]
    if n_components == 3:
        values, vectors = eigh(covar_matrix, eigvals=(d - 3, d - 1), eigvals_only=False)
        projected_data = np.dot(data, vectors)
        return projected_data[:, -1], projected_data[:, -2], projected_data[:, -3]
    values, vectors = eigh(covar_matrix, eigvals=(d - 2, d - 1), eigvals_only=False)
    projected_data = np.dot(data, vectors)
    return projected_data[:, -1], projected_data[:, -2]

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
            return data[:, 0], data[:, 1], np.zeros(data.shape[0])
        return data[:, 0], data[:, 1]
    if data.shape[1] < 2:
        return data[:, 0], data[:, 0]
    if dim == 2:
        if type_anal == 'pca':
            return get_pc_components(data)
        elif type_anal == 'random':
            random_dims  = random.sample(set(list(range(0, data.shape[1]))), dim)
            return data[0:data.shape[0], random_dims[0]], data[0:data.shape[0], random_dims[1]]
    elif dim == 3:
        if type_anal == 'pca':
            return get_pc_components(data, n_components=3)
        elif type_anal == 'random':
            random_dims  = random.sample(set(list(range(0, data.shape[1]))), dim)
            return data[0:data.shape[0], random_dims[0]], data[0:data.shape[0], random_dims[1]], data[0:data.shape[0], random_dims[2]]
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

def calc_spherical_mean_width_v2(this_data, num_directions=1e3):
    directions = np.random.rand(int(num_directions), this_data.shape[1]) # T x D
    directions = np.array(directions, dtype=np.double)
    # directions /= np.sqrt((directions ** 2).sum(-1))[..., np.newaxis]
    directions /= np.sqrt(np.einsum('...i,...i', directions, directions))[..., np.newaxis]
    mean_width = np.matmul(this_data, directions.transpose()) # N x T

    mean_width = np.max(mean_width, axis=0) - np.min(mean_width, axis=0)
    return np.mean(mean_width)


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
    result_dict['stable_rank'].append(np.sum(singular_values) / np.max(singular_values))
    result_dict['simple_spherical_mean_width'].append(2 * np.mean(singular_values))
    result_dict['spherical_mean_width_v2'].append(calc_spherical_mean_width_v2(this_data))
    
    # cdist(this_data, np.array([np.zeros(this_data.shape[1])]))
    result_dict['norms'].append(distance_from_origin(this_data))
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

def projection_plots(list_pca_2d, list_pca_3d, list_random_2d, list_random_3d, pre_path, costume_range=10):
    plot_data_projection(list_pca_2d, type_analysis='pca', dim=2, save_path='data_pca_2d.gif', pre_path=pre_path, costume_range=costume_range)
    plot_data_projection(list_pca_3d, type_analysis='pca', dim=3, save_path='data_pca_3d.gif', pre_path=pre_path, costume_range=costume_range)
    plot_data_projection(list_random_2d, type_analysis='random', dim=2, save_path='data_random_2d.gif', pre_path=pre_path, costume_range=costume_range)
    plot_data_projection(list_random_3d, type_analysis='random', dim=3, save_path='data_random_3d.gif', pre_path=pre_path, costume_range=costume_range)


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