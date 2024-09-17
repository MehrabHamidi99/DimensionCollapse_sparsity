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



def visualize_1D_boundaries(model, input_range=(-3, 3)):
    '''
    Parameters:
    model (torch.nn.Module): A trained PyTorch neural network model. 
    The model should have a single input neuron and a single output neuron.
    input_range (tuple, optional): A tuple defining the range of input values to visualize. Default is (-3, 3).
    
    Functionality:
    The function performs the following steps:
    Model Evaluation Mode: Sets the model to evaluation mode using model.eval().
    Input Sampling: Creates a dense, linearly spaced array of input values within the specified input_range.
    Model Inference: Passes the sampled input values through the neural network model.
    Gradient Computation:
    Computes the first and second order gradients of the model's output with respect to the input.
    Identifies potential ReLU activation boundaries based on the second-order gradient.
    Plotting:
    Plots the neural network's output as a function of the input.
    Highlights areas where the second-order gradient exceeds a certain threshold, indicating potential ReLU boundaries.
    Visualization:
    The function plots the output of the neural network for the sampled input range.
    Potential ReLU activation boundaries are marked in red on the plot.
    
    Usage Example:
    model = MLP_ReLU(n_in=1, layer_list=[10, 10, 1])
    visualize_1D_boundaries(model)

    Notes:
    The function is specifically designed for 1D input neural networks.
    It is a useful tool for understanding the decision-making process and activation dynamics of a network.
    The choice of boundary_threshold can be adjusted based on specific requirements.
    In the code, it is hard-coded to 0.001, but you can modify it to use the 95th percentile of 
    the absolute value of the second-order gradient or any other heuristic.
    This function provides a visual interpretation of how a one-dimensional neural network processes its input, 
    particularly highlighting the regions where ReLU activations switch on or off. It is valuable for 
    analyzing and understanding the behavior of simple neural network models.
    
    '''
    # Define the neural network
    model.eval()

    # Create a dense sampling of the 1D input space
    x = np.linspace(input_range[0], input_range[1], 10000).astype(np.double)
    x_tensor = torch.tensor(torch.DoubleTensor(x).unsqueeze(1), dtype=torch.double)

    # Pass samples through the network
    with torch.no_grad():
        model.double() #################### important
        y = model(x_tensor.double())

    y_grad = np.gradient(y.flatten(), x, edge_order=2)
    y_grad2 = np.gradient(y_grad, x, edge_order=2)

    # A high gradient indicates a likely boundary
    boundary_threshold = np.maximum(np.percentile(np.abs(y_grad2), 95), 0)  # e.g., 95th percentile
    boundary_threshold = 0.001
    # Plotting
    plt.plot(x, y, '-b', label='NN Output')
    plt.scatter(x[np.abs(y_grad2) > boundary_threshold], y[np.abs(y_grad2) > boundary_threshold],
                color='red', s=10, label='ReLU Boundaries')
    plt.legend()
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()

def animate_histogram(ax, counter, activation_data, title, name_fig='', x_axis_title='Activation Value', save_path='activation_animation.gif', bins=100, fps=1, pre_path='', fixed_scale=False, custom_range=20, step=False, zero_one=True, eigens=False, num=1000, norms=False):
    ax.clear()
    new_act = activation_data[counter]
    if zero_one:
        new_act = new_act / int(num)
    if eigens:    
        new_act = new_act[new_act > -1000]
    if norms:
        new_act = np.array(new_act)
        new_act /= max(1, np.max(new_act))

    if pd.isna(activation_data[counter]).any():
        if step:
            ax.hist(np.nan_to_num(new_act, nan=0), bins=bins, histtype='step')
        else:
            ax.hist(np.nan_to_num(new_act, nan=0).flatten(), bins=bins)
    else:
        if step:
            ax.hist(new_act.flatten(), bins=bins, histtype='step')
        else:
            ax.hist(new_act, bins=bins)
            plt.show()

    if type(title) is list:
        ax.set_title(title[counter])
    else:
        ax.set_title(f'{title} {counter + 1}')
    if zero_one:
        ax.set_xlim(0, 1)
    if fixed_scale:
        ax.set_xlim(0, custom_range + 0.1)
        # plt.xticks(np.arange(0, custom_range + 0.1, step=0.2))
    ax.set_xlabel(x_axis_title)
    ax.set_ylabel('Frequency')




def plot_gifs(result_dict, this_path, num, costume_range=None, pre_path=None, scale=None, eigenvectors=None, labels=None):

    global FRAMES_G
    FRAMES_G = 0

    layer_activation_ratio, eigens, dist_all, list_pca_2d, list_pca_3d, list_random_2d, list_random_3d =\
          result_dict['activations'], result_dict['eigenvalues'], result_dict['norms'], result_dict['pca_2'], result_dict['pca_3'], result_dict['random_2'], result_dict['random_3']
    dist_all = np.array(dist_all)

    
    fig, axs = plt.subplots(2, 4, figsize=(22, 22))  # Adjust subplot layout as needed


    def update(frames):
        global FRAMES_G
        animate_histogram(axs[0, 0], min(FRAMES_G, len(layer_activation_ratio) - 1), layer_activation_ratio, 'layer ', zero_one=True, num=num)
        animate_histogram(axs[0, 1], FRAMES_G, eigens, 'layers: ', x_axis_title='eigenvalues distribution', fixed_scale=False, custom_range=1, zero_one=False, eigens=True)  
        animate_histogram(axs[0, 2], FRAMES_G, dist_all, 'layers: ', x_axis_title='distance from origin distribution / max', fixed_scale=True, custom_range=1, step=False, zero_one=False, norms=True)

        plot_data_projection(axs[1, 0], FRAMES_G, list_pca_2d, type_analysis='pca', dim=2, costume_range=np.max(np.concatenate(list_pca_2d).ravel().tolist()), eigenvectors=eigenvectors, labels=labels)
        plot_data_projection(axs[1, 1], FRAMES_G, list_random_2d, type_analysis='random', dim=2, costume_range=np.max(np.concatenate(list_random_2d).ravel().tolist()), labels=labels)

        axs[1, 2].remove()
        axs[1, 2] = fig.add_subplot(2, 4, 7, projection='3d')
        plot_data_projection(axs[1, 2], FRAMES_G, list_pca_3d, type_analysis='pca', dim=3, costume_range=np.max(np.concatenate(list_pca_3d).ravel().tolist()), eigenvectors=eigenvectors, labels=labels)
        axs[1, 3].remove()
        axs[1, 3] = fig.add_subplot(2, 4, 8, projection='3d')
        plot_data_projection(axs[1, 3], FRAMES_G, list_random_3d, type_analysis='random', dim=3, costume_range=np.max(np.concatenate(list_random_3d).ravel().tolist()), labels=labels)
        FRAMES_G += 1

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(layer_activation_ratio), repeat=False)

    # Save the animation
    plt.grid(True)
    try:
        anim.save(pre_path + "all_gifa.gif", writer='imagemagick', fps=1)
    except RuntimeError:
        # print("Imagemagick writer not found. Falling back to Pillow writer.")
        try:
            anim.save(pre_path + "all_gifa.gif", writer=PillowWriter(fps=1))
        except Exception:
            # anim.save(pre_path + name_fig + save_path, writer=PillowWriter(fps=fps))
            pass
    plt.close()  # Close the plot to prevent it from displaying statically


def plot_data_projection(ax, counter, anim_pieces, type_analysis='pca', dim=2, title='layers: ', name_fig='', save_path='activation_animation.gif', bins=20, fps=1, pre_path='', costume_range=None, eigenvectors=None, labels=None):
    # eigenvectors_2d = eigenvectors[0]
    ax.clear()
    
    # Find unique labels and the number of unique classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Dynamically select an appropriate colormap
    if num_classes <= 10:
        cmap = plt.get_cmap('tab10')  # Use tab10 for 10 or fewer classes
    elif num_classes <= 20:
        cmap = plt.get_cmap('tab20')  # Use tab20 for 11-20 classes
    else:
        cmap = plt.cm.get_cmap('viridis', num_classes)  # Use continuous colormap for more than 20 classes
    
    # Map each unique label to a specific color in the colormap
    color_map = {label: cmap(i / num_classes) for i, label in enumerate(unique_labels)}
    
    # Assign colors to each data point based on its label
    colors = [color_map[label] for label in labels]
    
    # Select appropriate scatter plot based on dimension
    if dim == 2:
        ax.scatter(anim_pieces[counter][0], anim_pieces[counter][1], c=colors, s=10)
    elif dim == 3:
        ax.scatter(anim_pieces[counter][0], anim_pieces[counter][1], anim_pieces[counter][2], c=colors, s=10)
    
    # If labels are provided, add a legend
    if labels is not None:
        # Create a list of legend handles
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label], 
                              markersize=8, label=f'Class {label}') for label in unique_labels]
        
        # Add the legend to the axis
        ax.legend(handles=handles, title="Classes", loc="best")
    
    ax.grid(True)

    if type_analysis == 'pca':
        ax.set_xlabel('First Principal Component')
        ax.set_ylabel('Second Principal Component')
        if dim == 2:
            # ax.quiver([0, 0], eigenvectors[0][:, -1][0], eigenvectors[0][:, -1][1])
            # ax.quiver([0, 0], eigenvectors[0][:, -2][0], eigenvectors[0][:, -2][1])
            ax.set_title('Data in First Two Principal Components')
        if dim == 3:
            ax.set_zlabel('Third Principal Component')
            ax.set_title('Data in First Three Principal Components')
    elif type_analysis == 'random':
        ax.set_xlabel('First Random Diemnsion')
        ax.set_ylabel('Second Random Dimension')
        ax.set_title('Data in Two Random Dimension')
        if dim == 2:
            ax.set_title('Data in Two Random Dimension')
        if dim == 3:
            ax.set_zlabel('Third Random Dimension')
            ax.set_title('Data in Three Random Dimension')
    if type(title) is list:
        ax.set_title(title[counter])
    else:
        ax.set_title(f'{title} {counter + 1}')
    # if costume_range:
    #     ax.set_ylim(-1 * costume_range, costume_range)
    #     ax.set_xlim(-1 * costume_range, costume_range)
    #     if dim == 3:
    #         ax.set_zlim(-1 * costume_range, costume_range)
        
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

def plotting_actions(result_dict, num, this_path, arch, suffix=''):

    activations, stable_ranks_all, simple_spherical_mean_width_all, spherical_mean_width_v2_all, eigen_count, distances, display_neuron_matrx, cell_dims = result_dict['activations'], result_dict['stable_rank'], result_dict['simple_spherical_mean_width'], result_dict['spherical_mean_width_v2'], result_dict['nonzero_eigenvalues_count'], result_dict['norms'], result_dict['display_matrix'], result_dict['batch_cell_dimensions']

    act_count = np.array(list(itertools.chain.from_iterable(list(activations))))
    fig, ax = plt.subplots(4, 2, figsize=(20, 20))
    if isinstance(arch, str):
        fig.suptitle('Neuron activation Frequency{}, cell dim: {}'.format(arch, str(np.max(cell_dims))))
    else:
        fig.suptitle('#neurons:{}, #layers:{}'.format(str(np.sum(arch)), str(len(arch)), str(np.max(cell_dims))))


    # Activation plot
    tp = ax[0, 0].hist(act_count / num, bins=100)
    ax[0, 0].set_xlabel('neuron activation percentage of a given dataset')
    ax[0, 0].set_ylabel('neuron frequency')
    ax[0, 0].set_title('Single Neuron Activations')
    # fig.savefig(this_path + suffix + 'additive_activations.png')
    # plt.xlim(0, 1)
    # plt.close(fig)

    # Eigenvalue plots:
    # fig, ax = plt.subplots(figsize=(7, 7))
    tp = ax[0, 1].bar(np.arange(len(eigen_count)), eigen_count)
    ax[0, 1].set_ylabel('number of non-zero eigenvalues')
    ax[0, 1].set_xlabel('layers')
    ax[0, 1].set_title('Non-zero eigenvalues')
    # fig.savefig(this_path + suffix + 'non_zero_eigenvalues.png')
    # plt.close(fig)


# def plot_distances(net, distances, this_path, suffix=''):
    # fig, ax = plt.subplots(figsize=(7, 7))
    X_axis = np.arange(len(distances))
    dis_stat = np.array(distances)

    ax[1, 0].bar(X_axis - 0.2, np.mean(distances, axis=1), 0.2, label = 'Mean')
    # ax[1, 0].bar(X_axis + 0.0, np.var(distances, axis=1), 0.2, label = 'Var')
    ax[1, 0].bar(X_axis + 0.0, np.max(distances, axis=1), 0.2, label = 'Max')
    # plt.bar(X_axis + 0.2, dis_stat[:, 1] / dis_stat[:, 0], 0.2, label = 'Max / mean')
    ax[1, 0].set_ylabel('mean distances from origin')
    ax[1, 0].set_xlabel('layers')
    ax[1, 0].set_title('Max/Mean Norms throughout layers')
    ax[1, 0].legend()
    
    sns.heatmap(display_neuron_matrx, cmap="mako", annot=False, ax=ax[1, 1])
    ax[1, 1].set_title('Neuron activation heatmap')


    tp = ax[2, 0].bar(np.arange(len(stable_ranks_all)), stable_ranks_all)
    ax[2, 0].set_ylabel('Stable Rank')
    ax[2, 0].set_xlabel('layers')
    ax[2, 0].set_title('Stable Rank throughout layers')

    tp = ax[2, 1].bar(np.arange(len(simple_spherical_mean_width_all)), simple_spherical_mean_width_all)
    ax[2, 1].set_ylabel('Spherical Mean Width')
    ax[2, 1].set_xlabel('layers')
    ax[2, 1].set_title('2 * Mean singular values')

    tp = ax[3, 0].bar(np.arange(len(spherical_mean_width_v2_all)), spherical_mean_width_v2_all)
    ax[3, 0].set_ylabel('Spherical Mean Width - v2')
    ax[3, 0].set_xlabel('layers')
    ax[3, 0].set_title('Spherical Mean Width')

    tp = ax[3, 1].bar(np.arange(len(cell_dims)), cell_dims)
    ax[3, 1].set_ylabel('dimensions')
    ax[3, 1].set_xlabel('layers')
    ax[3, 1].set_title('Cell Dimensions')

    fig.savefig(this_path + suffix + 'all_plots.png')
    plt.close(fig)

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

def batch_projectional_analysis(covariance_matrix, data, result_dict, first_batch, this_index=0, preds=None):

    def get_projected_data(result_dict, projected_data, data):
        if data.shape[1] > 3:
            result_dict['pca_2'][this_index][0].extend(projected_data[:, -1])
            result_dict['pca_2'][this_index][1].extend(projected_data[:, -2])
            
            result_dict['pca_3'][this_index][0].extend(projected_data[:, -1])
            result_dict['pca_3'][this_index][1].extend(projected_data[:, -2])
            result_dict['pca_3'][this_index][2].extend(projected_data[:, -3])

            random_dims  = random.sample(set(list(range(0, data.shape[1]))), 2)
            result_dict['random_2'][this_index][0].extend(data[0:data.shape[0], random_dims[0]])
            result_dict['random_2'][this_index][1].extend(data[0:data.shape[0], random_dims[1]])

            random_dims  = random.sample(set(list(range(0, data.shape[1]))), 3)
            result_dict['random_3'][this_index][0].extend(data[0:data.shape[0], random_dims[0]])
            result_dict['random_3'][this_index][1].extend(data[0:data.shape[0], random_dims[1]])
            result_dict['random_3'][this_index][2].extend(data[0:data.shape[0], random_dims[2]])
        elif data.shape[1] == 3:

            result_dict['pca_2'][this_index][0].extend(projected_data[:, -1])
            result_dict['pca_2'][this_index][1].extend(projected_data[:, -2])

            random_dims  = random.sample(set(list(range(0, data.shape[1]))), 2)
            result_dict['random_2'][this_index][0].extend(data[:, 0])
            result_dict['random_2'][this_index][1].extend(data[:, 1])

            result_dict['pca_3'][this_index][0].extend(data[:, 0])
            result_dict['pca_3'][this_index][1].extend(data[:, 1])
            result_dict['pca_3'][this_index][2].extend(data[:, 2])

            result_dict['random_3'][this_index][0].extend(data[:, 0])
            result_dict['random_3'][this_index][1].extend(data[:, 1])
            result_dict['random_3'][this_index][2].extend(data[:, 2])
            
        elif data.shape[1] == 2:
            result_dict['pca_2'][this_index][0].extend(data[:, 0])
            result_dict['pca_2'][this_index][1].extend(data[:, 1])

            result_dict['random_2'][this_index][0].extend(data[:, 0])
            result_dict['random_2'][this_index][1].extend(data[:, 1])

            result_dict['pca_3'][this_index][0].extend(data[:, 0])
            result_dict['pca_3'][this_index][1].extend(data[:, 1])
            result_dict['pca_3'][this_index][2].extend(data[:, 1] * 0)

            result_dict['random_3'][this_index][0].extend(data[:, 0])
            result_dict['random_3'][this_index][1].extend(data[:, 1])
            result_dict['random_3'][this_index][2].extend(data[:, 1] * 0)
        else:
            result_dict['pca_2'][this_index][0].extend(data[:, 0])
            result_dict['pca_2'][this_index][1].extend(data[:, 0] * 0)

            result_dict['random_2'][this_index][0].extend(data[:, 0])
            result_dict['random_2'][this_index][1].extend(data[:, 0] * 0)

            result_dict['pca_3'][this_index][0].extend(data[:, 0])
            result_dict['pca_3'][this_index][1].extend(data[:, 0] * 0)
            result_dict['pca_3'][this_index][2].extend(data[:, 0] * 0)

            result_dict['random_3'][this_index][0].extend(data[:, 0])
            result_dict['random_3'][this_index][1].extend(data[:, 0])
            result_dict['random_3'][this_index][2].extend(data[:, 0] * 0)

        return result_dict

    covar_matrix = covariance_matrix.cpu().detach().numpy()

    values, vectors = eigh(covar_matrix, eigvals_only=False)

    projected_data = np.matmul(data, vectors)

    if first_batch:
        result_dict['pca_2'].append(([], []))
        result_dict['pca_3'].append(([], [], []))
        result_dict['random_2'].append(([], []))
        result_dict['random_3'].append(([], [], []))
        result_dict['norms'].append([])
        result_dict['spherical_mean_width_v2'].append(0)

    # if preds is not None:
    #     result_dict['labels'].extend(preds)
        
    result_dict['spherical_mean_width_v2'][this_index] = (result_dict['spherical_mean_width_v2'][this_index] + calc_spherical_mean_width_v2(data)) / 2.0

    result_dict['norms'][this_index].extend(distance_from_origin(data).tolist())

    result_dict = get_projected_data(result_dict, projected_data, data)

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



def create_gif_from_plots(plot_dir, output_gif, plot_type, start_epoch=10, end_epoch=100, step=10, gif_duration=500):
    frames = []
    font = ImageFont.load_default()  # You can replace this with a path to a font file if you need a custom font

    for epoch in range(start_epoch, end_epoch + 1, step):
        # Load both train and validation images for each epoch
        img_path = os.path.join(plot_dir, f'epoch_{epoch}/_{plot_type}_all_plots.png')

        # Open the image (train or validation plot)
        img = Image.open(img_path)

        # Add epoch title to the plot
        draw = ImageDraw.Draw(img)
        text = f"Epoch {epoch} - {plot_type.capitalize()}"
        text_width, text_height = draw.textsize(text, font=font)
        draw.text(((img.width - text_width) // 2, 10), text, font=font, fill="white")

        # Append the frame to the list
        frames.append(img)

    # Save the frames as a GIF
    frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=gif_duration, loop=0)
