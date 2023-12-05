import pandas as pd
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.linear_model import LinearRegression, RANSACRegressor, BayesianRidge
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os
# os.chdir('/content/gdrive/MyDrive/DeepReluSymmetries/')
import seaborn as sns
import pandas as pd


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

    # v_grad = np.vectorize(compute_gradient_vmap, signature='(n)->(n, 1)', excluded={0})
    # grads = v_grad(model, x)
    # print(grads)
    # print(y_grad)

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

def animate_histogram(activation_data, title, name_fig='', x_axis_title='Activation Value', save_path='activation_animation.gif', bins=20, fps=1, pre_path=''):
    fig, ax = plt.subplots(figsize=(8, 6))
    def update(counter):
        ax.clear()
        if pd.isna(activation_data[counter]).any():
            ax.hist(np.nan_to_num(activation_data[counter], nan=0), bins=bins)
        else:
            ax.hist(activation_data[counter], bins=bins)
        if type(title) is list:
            ax.set_title(title[counter])
        else:
            ax.set_title(f'{title} {counter + 1}')
        ax.set_xlabel(x_axis_title)
        ax.set_ylabel('Frequency')

    anim = FuncAnimation(fig, update, frames=len(activation_data), repeat=False)

    try:
        anim.save(pre_path + name_fig + save_path, writer='imagemagick', fps=fps)
    except RuntimeError:
        print("Imagemagick writer not found. Falling back to Pillow writer.")
        try:
            anim.save(pre_path + name_fig + save_path, writer=PillowWriter(fps=fps))
        except Exception:
            # anim.save(pre_path + name_fig + save_path, writer=PillowWriter(fps=fps))
            pass
    plt.close(fig)
    return anim


def plot_data_with_pca(data, n_components=2):
    """
    Plots data in the first two principal component dimensions after performing PCA.

    Parameters:
    data (numpy.ndarray): A 2D numpy array where rows represent samples and columns represent features.

    """

    # Perform PCA
    pca = PCA(n_components=n_components)
    pcs = pca.fit_transform(data)

    # Plot the data in the first two principal components
    # plt.figure(figsize=(8, 6))
    # plt.scatter(pcs[:, 0], pcs[:, 1], alpha=0.7)
    # plt.xlabel('First Principal Component')
    # plt.ylabel('Second Principal Component')
    # plt.title('Data in First Two Principal Components')
    # plt.grid(True)
    # plt.show()

    # Print eigenvalues
    # print("Eigenvalues of the first two principal components:", pca.explained_variance_)

    return pcs[:, 0], pcs[:, 1]

def plot_data_pca_animation(data, model, name_fig='', save_path='activation_animation.gif', bins=20, fps=1, pre_path=''):
    all_pcs = model.plot_data_animation(data)
    N_plots = len(all_pcs)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    
    def update(counter):
        ax.clear()
        ax.scatter(all_pcs[counter][0], all_pcs[counter][1], alpha=0.7)

        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.title('Data in First Two Principal Components')

    anim = FuncAnimation(fig, update, frames=N_plots, repeat=False)

    try:
        anim.save(pre_path + name_fig + save_path, writer='imagemagick', fps=fps)
    except RuntimeError:
        print("Imagemagick writer not found. Falling back to Pillow writer.")
        try:
            anim.save(pre_path + name_fig + save_path, writer=PillowWriter(fps=fps))
        except Exception:
            # anim.save(pre_path + name_fig + save_path, writer=PillowWriter(fps=fps))
            pass

    plt.grid(True)
    plt.close(fig)
    return anim

def count_near_zero_eigenvalues(data, threshold=0.001, return_eigenvalues=False):
    """
    Counts the number of eigenvalues of the covariance matrix of the data that are close to zero.

    Parameters:
    data (numpy.ndarray): A 2D numpy array where rows represent samples and columns represent features.
    threshold (float): A threshold to consider an eigenvalue as 'close to zero'. Default is 0.01.

    Returns:
    int: The number of eigenvalues close to zero.
    """

    # Perform PCA
    pca = PCA()
    pca.fit(data)

    # Eigenvalues
    eigenvalues = pca.explained_variance_

    # Count eigenvalues close to zero
    near_zero_count = np.sum(np.abs(eigenvalues) > threshold)
    if return_eigenvalues:
        return near_zero_count, eigenvalues
    return near_zero_count


def file_name_handling(which, architecture, num, exps=1, pre_path='', normal_dist=False, loc=0, scale=1):
    if normal_dist:
        pre_path += 'normal_std{}/'.format(str(scale))
    else:
        pre_path += 'uniform/'
    this_path = pre_path + which + '_{}_exps{}_num{}/'.format(str(architecture), str(exps), str(num))
    if not os.path.isdir(this_path):
        try:
            os.makedirs(this_path)
        except OSError as exc:
            this_path = pre_path + which + '_{}_{}_{}_exps{}_num{}/'.format(str(architecture[0]), str(architecture[1][0]), str(len(architecture[1])), str(exps), str(num))
            if not os.path.isdir(this_path):
                os.makedirs(this_path)
    return this_path

def plotting_actions(res1, eigen_count, num, this_path, net, suffix=''):
    # Activation plot
    fig, ax = plt.subplots(figsize=(10, 10))
    tp = ax.hist(res1 / num, bins=20)
    ax.set_xlabel('neuron activation percentage of a given dataset')
    ax.set_ylabel('neuron frequency')
    ax.set_title('#neurons:{}, #layers:{}'.format(str(np.sum(net.layer_list)), str(len(net.layer_list))))
    fig.savefig(this_path + suffix + 'additive_activations.pdf')
    plt.close(fig)

    # Eigenvalue plots:
    fig, ax = plt.subplots(figsize=(10, 10))
    tp = ax.bar(np.arange(len(eigen_count)), eigen_count)
    ax.set_ylabel('number of non-zero eigenvalues')
    ax.set_xlabel('layers')
    ax.set_title('Non-zero eigenvalues for network with #neurons:{}, #layers:{}'.format(str(np.sum(net.layer_list)), str(len(net.layer_list))))
    fig.savefig(this_path + suffix + 'non_zero_eigenvalues.pdf')
    plt.close(fig)




# if normal_dist:
#     pre_path += 'normal_std{}/'.format(str(scale))
#   else:
#     pre_path += 'uniform/'
#   this_path = pre_path + 'random_data_random_untrained_network{}_exps{}_num{}/'.format(str(architecture), str(exps), str(num))
#   if not os.path.isdir(this_path):
#     try:
#       os.makedirs(this_path)
#     except OSError as exc:
#       this_path = pre_path + 'random_data_random_untrained_network{}_{}_{}_exps{}_num{}/'.format(str(architecture[0]), str(architecture[1][0]), str(len(architecture[1])), str(exps), str(num))
#       if not os.path.isdir(this_path):
#         os.makedirs(this_path)