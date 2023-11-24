import pandas as pd
import numpy as np

from collections import defaultdict

import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression, RANSACRegressor, BayesianRidge

import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import os
# os.chdir('/content/gdrive/MyDrive/DeepReluSymmetries/')
import seaborn as sns

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
    print(boundary_threshold)
    boundary_threshold = 0.001
    # Plotting
    plt.plot(x, y, '-b', label='NN Output')
    plt.scatter(x[np.abs(y_grad2) > boundary_threshold], y[np.abs(y_grad2) > boundary_threshold],
                color='red', s=10, label='ReLU Boundaries')
    plt.legend()
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.show()

def animate_histogram(activation_data, title, name_fig='', save_path='activation_animation.gif', bins=20, fps=1, pre_path=''):
    fig, ax = plt.subplots()

    def update(counter):
        ax.clear()
        ax.hist(activation_data[counter], bins=bins)
        ax.set_title(f'{title} {counter + 1}')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')

    anim = FuncAnimation(fig, update, frames=len(activation_data), repeat=False)

    try:
        anim.save(pre_path + name_fig + save_path, writer='imagemagick', fps=fps)
    except RuntimeError:
        print("Imagemagick writer not found. Falling back to Pillow writer.")
        anim.save(pre_path + save_path, writer=PillowWriter(fps=fps))
    plt.show()
    plt.close(fig)
    return anim