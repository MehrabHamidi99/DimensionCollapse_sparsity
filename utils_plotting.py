from utils import *
from utils import calc_spherical_mean_width_v2
import pandas as pd
import numpy as np
np.random.seed(33)
from scipy.spatial.distance import pdist, squareform
from scipy.spatial.distance import cdist
from scipy.linalg import eigh

from collections import defaultdict

import os
from PIL import Image

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

import gc


def animate_histogram(ax, counter, activation_data, title, name_fig='', x_axis_title='Activation Value', save_path='activation_animation.gif', bins=100, fps=1, pre_path='', fixed_scale=False, custom_range=20, step=False, zero_one=True, eigens=False, num=1000, norms=False):
    ax.cla()
    new_act = np.array(activation_data[counter])
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



def calculate_custom_range(data_list, dim, index=None, fixed=True):
    """
    Calculate the custom range for the axes based on the maximum values in the data list.
    
    Parameters:
    - data_list: List of lists of data arrays.
    - dim: Dimension of the data (2 or 3).
    
    Returns:
    - custom_range: Dictionary with keys 'x', 'y', and optionally 'z' for custom ranges.
    """

    if index:
        # Flatten the list of lists
        all_data = np.array(data_list)
        
        custom_range = {
            'x': [np.min(all_data[index, 0, :]), np.max(all_data[index, 0, :])],
            'y': [np.min(all_data[index, 1, :]), np.max(all_data[index, 1, :])]
        }
        if dim == 3:
            custom_range['z'] = [np.min(all_data[index, 2, :]), np.max(all_data[index, 2, :])]

        print(custom_range)

        if fixed:
            custom_range['x'][0] = np.min(custom_range['x'][0], custom_range['y'][0], custom_range['z'][0])
            custom_range['y'][0] = np.min(custom_range['x'][0], custom_range['y'][0], custom_range['z'][0])

            custom_range['x'][1] = np.max(custom_range['x'][1], custom_range['y'][1], custom_range['z'][1])
            custom_range['y'][1] = np.max(custom_range['x'][1], custom_range['y'][1], custom_range['z'][1])

            if dim == 3:
                custom_range['z'][0] = np.min(custom_range['x'][0], custom_range['y'][0], custom_range['z'][0])
            custom_range['z'][1] = np.max(custom_range['x'][1], custom_range['y'][1], custom_range['z'][1])

        return custom_range

    # Flatten the list of lists
    all_data = np.array(data_list)
    
    custom_range = {
        'x': [np.min(all_data[:, 0, :]), np.max(all_data[:, 0, :])],
        'y': [np.min(all_data[:, 1, :]), np.max(all_data[:, 1, :])]
    }

    if dim == 3:
        custom_range['z'] = [np.min(all_data[:, 2, :]), np.max(all_data[:, 2, :])]
    
    if fixed:
        custom_range['x'][0] = np.min(custom_range['x'][0], custom_range['y'][0], custom_range['z'][0])
        custom_range['y'][0] = np.min(custom_range['x'][0], custom_range['y'][0], custom_range['z'][0])

        custom_range['x'][1] = np.max(custom_range['x'][1], custom_range['y'][1], custom_range['z'][1])
        custom_range['y'][1] = np.max(custom_range['x'][1], custom_range['y'][1], custom_range['z'][1])

        if dim == 3:
            custom_range['z'][0] = np.min(custom_range['x'][0], custom_range['y'][0], custom_range['z'][0])
        custom_range['z'][1] = np.max(custom_range['x'][1], custom_range['y'][1], custom_range['z'][1])

    print(custom_range)
    return custom_range


def plot_gifs(result_dict, this_path, num, custom_range=None, pre_path: str = '', scale=None, eigenvectors=None, labels: list = [], layer_names: list = [], no_custome_range='all'):

    layer_activation_ratio, eigens, dist_all, list_pca_2d, list_pca_3d, list_random_2d, list_random_3d =\
          result_dict['activations'], result_dict['eigenvalues'], result_dict['norms'], result_dict['pca_2'], result_dict['pca_3'], result_dict['random_2'], result_dict['random_3']
    dist_all = np.array(dist_all)

    # Determine grid size to make a square-like grid based on len(list_pca_2d)
    num_frames = len(list_pca_2d)
    arg_divisor = np.array([[i, num_frames // i] for i in range(1, num_frames) if num_frames % i == 0 and i <= num_frames // i])
    largest_divisor = arg_divisor[np.argmin(arg_divisor[:, 1] - arg_divisor[:, 0]), 0]
    # largest_divisor = max([i for i in range(1, num_frames) if num_frames % i == 0])
    if largest_divisor == 1:
        largest_divisor = num_frames
    grid_size = (num_frames // largest_divisor, largest_divisor)

    os.makedirs(pre_path + 'all_gifs/', exist_ok=True)
    os.makedirs(pre_path + 'all_gifs_pdf/', exist_ok=True)


    if no_custome_range == 'all':
        # Calculate custom ranges once for all frames
        custom_range_2d = calculate_custom_range(list_pca_2d, dim=2)
        custom_range_2d_random = calculate_custom_range(list_random_2d, dim=2)
        custom_range_3d = calculate_custom_range(list_pca_3d, dim=3)
        custom_range_3d_random = calculate_custom_range(list_random_3d, dim=3)
    elif no_custome_range == 'input':
        # Calculate custom ranges once for all frames
        custom_range_2d = calculate_custom_range(list_pca_2d, dim=2, index=0)
        custom_range_2d_random = calculate_custom_range(list_random_2d, dim=2, index=0)
        custom_range_3d = calculate_custom_range(list_pca_3d, dim=3, index=0)
        custom_range_3d_random = calculate_custom_range(list_random_3d, dim=3, index=0)
    else:
        custom_range = None
        custom_range_2d = None
        custom_range_2d_random = None
        custom_range_3d = None
        custom_range_3d_random = None



    # Create a list to store the axes for each frame
    for frame in range(num_frames):
        fig, axs = plt.subplots(2, 4, figsize=(22, 22))  # Adjust subplot layout as needed

        if len(layer_names) > 0:
            subplot_title = layer_names
        else:
            subplot_title = 'layers: '
        
        animate_histogram(axs[0, 0], min(frame, len(layer_activation_ratio) - 1), layer_activation_ratio, title=subplot_title, zero_one=True, num=num)
        animate_histogram(axs[0, 1], frame, eigens, title=subplot_title, x_axis_title='eigenvalues distribution', fixed_scale=False, custom_range=1, zero_one=False, eigens=True)  
        animate_histogram(axs[0, 2], frame, dist_all, title=subplot_title, x_axis_title='distance from origin distribution / max', fixed_scale=True, custom_range=1, step=False, zero_one=False, norms=True)

        # plot_data_projection(axs[1, 0], frame, list_pca_2d, title=subplot_title, type_analysis='pca', dim=2, custom_range=np.max(np.concatenate(list_pca_2d).ravel().tolist()), eigenvectors=eigenvectors, labels_all=labels)
        # plot_data_projection(axs[1, 1], frame, list_random_2d, title=subplot_title, type_analysis='random', dim=2, custom_range=np.max(np.concatenate(list_random_2d).ravel().tolist()), labels_all=labels)
        plot_data_projection(axs[1, 0], frame, list_pca_2d, title=subplot_title, type_analysis='pca', dim=2, custom_range=custom_range_2d, eigenvectors=eigenvectors, labels_all=labels)
        plot_data_projection(axs[1, 1], frame, list_random_2d, title=subplot_title, type_analysis='random', dim=2, custom_range=custom_range_2d_random, labels_all=labels)
        
        axs[1, 2].remove()
        axs[1, 2] = fig.add_subplot(2, 4, 7, projection='3d')
        plot_data_projection(axs[1, 2], frame, list_pca_3d, title=subplot_title, type_analysis='pca', dim=3, custom_range=custom_range_3d, eigenvectors=eigenvectors, labels_all=labels)
        axs[1, 3].remove()
        axs[1, 3] = fig.add_subplot(2, 4, 8, projection='3d')
        plot_data_projection(axs[1, 3], frame, list_random_3d, title=subplot_title, type_analysis='random', dim=3, custom_range=custom_range_3d_random, labels_all=labels)
        
        fig.savefig(pre_path + "all_gifs_pdf/{}_layer.pdf".format(str(frame)))
        fig.savefig(pre_path + "all_gifs/{}_layer.png".format(str(frame)))
        plt.close(fig)
    
    # if num_frames % grid_size != 0:
    #     if num_frames > 10:
    #         raise Exception("Can't create grid")
    #     else:
    #         grid_size = num_frames

    # Create 8 figures, each containing all frames for one of the subplots
    for i in range(7):
        fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=(28 + grid_size[0], 15 + grid_size[1]))
        axs_ = axs.flatten()
        ax_3d = []

        for frame in range(num_frames):
            fig_ax = axs_[frame]

            if len(layer_names) > 0:
                subplot_title = layer_names
            else:
                subplot_title = 'layers: '

            # Re-generate only the specific subplot for the frame
            if i == 0:
                animate_histogram(fig_ax, min(frame, len(layer_activation_ratio) - 1), layer_activation_ratio, title=subplot_title, zero_one=True, num=num)
            elif i == 1:
                animate_histogram(fig_ax, frame, eigens, title=subplot_title, x_axis_title='eigenvalues distribution', fixed_scale=False, custom_range=1, zero_one=False, eigens=True)
            elif i == 2:
                animate_histogram(fig_ax, frame, dist_all, title=subplot_title, x_axis_title='distance from origin distribution / max', fixed_scale=True, custom_range=1, step=False, zero_one=False, norms=True)
            elif i == 3:
                plot_data_projection(fig_ax, frame, list_pca_2d, title=subplot_title, type_analysis='pca', dim=2, custom_range=custom_range_2d, eigenvectors=eigenvectors, labels_all=labels, add_legend=False)
            elif i == 4:
                plot_data_projection(fig_ax, frame, list_random_2d, title=subplot_title, type_analysis='random', dim=2, custom_range=custom_range_2d_random, labels_all=labels, add_legend=False)
            elif i == 5:
                axs_[frame].remove()
                # fig.delaxes(axs[1])
                fig_ax = fig.add_subplot(grid_size[0], grid_size[1], frame + 1, projection='3d')
                plot_data_projection(fig_ax, frame, list_pca_3d, title=subplot_title, type_analysis='pca', dim=3, custom_range=custom_range_3d, eigenvectors=eigenvectors, labels_all=labels, add_legend=False, font_size=10)
                ax_3d.append(fig_ax)
            elif i == 6:
                # axs_[frame].remove()
                axs_[frame].remove()
                fig_ax = fig.add_subplot(grid_size[0], grid_size[1], frame + 1, projection='3d')
                # axs[frame] = fig_ax
                plot_data_projection(fig_ax, frame, list_random_3d, title=subplot_title, type_analysis='random', dim=3, custom_range=custom_range_3d_random, labels_all=labels, add_legend=False, font_size=10)
                ax_3d.append(fig_ax)

            if len(layer_names) > 0:
                fig_ax.set_title(layer_names[frame])
            else:
                subplot_title = 'layers: '
                fig_ax.set_title(f'layers: {frame}')
            
            # fig_ax.axis('off')
        # Hide any empty subplots
        for j in range(num_frames, len(axs)):
            axs[j].axis('off')

        # Set axis labels only for the bottom row and left column, remove for others
        for ax_row in axs.reshape(grid_size)[:-1, :].flatten():
            ax_row.set_xlabel('')
            ax_row.set_xticks([])

        for ax_col in axs.reshape(grid_size)[:, 1:].flatten():
            ax_col.set_ylabel('')
            ax_col.set_yticks([])

            
        # for ax_row in axs.reshape(grid_size)[-1, :]:
        #     ax_row.set_xlabel('X-axis')
        # for ax_col in axs.reshape(grid_size)[:, 0]:
        #     ax_col.set_ylabel('Y-axis')

        fig.savefig(pre_path + "all_gifs_pdf/all_frames_subplot_{}.pdf".format(i))
        fig.savefig(pre_path + "all_gifs/all_frames_subplot_{}.png".format(i))
        plt.close(fig)
        
    # _create_gif_from_images(pre_path + 'all_gifs/', pre_path + 'all_gifs.gif')



def _create_gif_from_images(image_dir, output_gif_path, duration=1500):

    # Function to extract the layer index from the filename
    def get_layer_index(filename):
        base = os.path.basename(filename)
        # Split the filename to extract the index
        # Assuming filename is like '0_layer.png' or '.../0_layer.png'
        parts = base.split('_')
        # if 'layer' not in parts:
        #     return -1
        if parts and parts[0].isdigit():
            return int(parts[0])
        else:
            return -1  # Default value for sorting non-matching files


    # Get list of image file paths in the directory
    # image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('layer.pdf')]
    image_files = [os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith('layer.png')]

    # Sort the images based on the extracted layer index
    image_files_sorted = sorted(
        image_files,
        key=lambda x: get_layer_index(x)
    )

    images = [Image.open(img) for img in image_files_sorted]

    # Save as GIF
    images[0].save(
        output_gif_path,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0
    )

def plot_data_projection(ax, counter, anim_pieces, type_analysis='pca', dim=2, title='layers: ', name_fig='', save_path='activation_animation.gif', bins=20, fps=1, pre_path='', custom_range=None, eigenvectors=None, labels_all: list = [], new=False, add_legend=True, font_size=16):
    ax.cla()

    # if custom_range is None:
    #     custom_range = {'x': None, 'y': None, 'z': None}

    # # Default ranges
    # default_range = {'x': [min(data[:, 0]), max(data[:, 0])],
    #                  'y': [min(data[:, 1]), max(data[:, 1])]}
    
    # if data.shape[1] == 3:
    #     default_range['z'] = [min(data[:, 2]), max(data[:, 2])]

    # # Use default range if counter is zero
    # if counter == 0:
    #     custom_range = default_range
    # else:
    #     # Save the default range for future use
    #     if 'default_range' not in plot_data_projection.__dict__:
    #         plot_data_projection.default_range = default_range

    #     # Use saved default range if custom range is not provided
    #     custom_range = {axis: custom_range[axis] if custom_range[axis] is not None else plot_data_projection.default_range[axis]
    #                     for axis in custom_range}

    # Extract labels if provided
    if len(labels_all) == 0:
        labels = np.ones(len(anim_pieces[counter][0]))
    else:
        labels = labels_all[0]
    
    # Find unique labels and the number of unique classes
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    
    # Dynamically select an appropriate colormap
    if num_classes <= 10:
        cmap = plt.get_cmap('tab10')  # type: ignore # Use tab10 for 10 or fewer classes
    elif num_classes <= 20:
        cmap = plt.get_cmap('tab20')  # type: ignore # Use tab20 for 11-20 classes
    else:
        cmap = plt.cm.get_cmap('viridis', num_classes)  # type: ignore # Use continuous colormap for more than 20 classes
    
    # Map each unique label to a specific color in the colormap
    color_map = {label: cmap(i / num_classes) for i, label in enumerate(unique_labels)}
    
    # Assign colors to each data point based on its label
    colors = [color_map[label] for label in labels]
    
    # Select appropriate scatter plot based on dimension
    if dim == 2:
        ax.scatter(anim_pieces[counter][0], anim_pieces[counter][1], c=colors, s=10)
        if custom_range:
            ax.set_xlim(custom_range['x'][0], custom_range['x'][1])
            ax.set_ylim(custom_range['y'][0], custom_range['y'][1])
    elif dim == 3:
        ax.scatter(anim_pieces[counter][0], anim_pieces[counter][1], anim_pieces[counter][2], c=colors, s=10)
        if custom_range:
            ax.set_xlim(custom_range['x'][0], custom_range['x'][1])
            ax.set_ylim(custom_range['y'][0], custom_range['y'][1])
            ax.set_zlim(custom_range['z'][0], custom_range['z'][1])
    
    # If labels are provided, add a legend
    if labels is not None:
        if num_classes > 1:
            if add_legend:
                # Create a list of legend handles
                handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map[label],  # type: ignore
                                    markersize=8, label=f'Class {label}') for label in unique_labels]
                
                # Add the legend to the axis
                ax.legend(handles=handles, title="Classes", loc="best")
    
    ax.grid(True)

    if type_analysis == 'pca':
        ax.set_xlabel('First Principal Component', fontsize=font_size)
        ax.set_ylabel('Second Principal Component', fontsize=font_size)
        if dim == 2:
            # ax.quiver([0, 0], eigenvectors[0][:, -1][0], eigenvectors[0][:, -1][1])
            # ax.quiver([0, 0], eigenvectors[0][:, -2][0], eigenvectors[0][:, -2][1])
            ax.set_title('Data in First Two Principal Components')
        if dim == 3:
            ax.set_zlabel('Third Principal Component', fontsize=font_size)
            ax.set_title('Data in First Three Principal Components')
    elif type_analysis == 'random':
        ax.set_xlabel('First Random Diemnsion', fontsize=font_size)
        ax.set_ylabel('Second Random Dimension', fontsize=font_size)
        ax.set_title('Data in Two Random Dimension')
        if dim == 2:
            ax.set_title('Data in Two Random Dimension')
        if dim == 3:
            ax.set_zlabel('Third Random Dimension', fontsize=font_size)
            ax.set_title('Data in Three Random Dimension')
    if type(title) is list:
        ax.set_title(title[counter])
    else:
        ax.set_title(f'{title} {counter + 1}')

    return colors


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
    
    sns.heatmap(display_neuron_matrx, cmap="mako", annot=False, ax=ax[1, 1], vmin=0, vmax=num)
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

    fig.savefig(this_path + suffix + 'all_plots.pdf')
    plt.close(fig)

def batch_projectional_analysis(covariance_matrix, data, result_dict, first_batch, this_index=0, preds=None):

    def get_projected_data(result_dict, projected_data, data):
        if data.shape[1] > 3:
            result_dict['pca_2'][this_index][0].extend(projected_data[:, -1].cpu().detach().numpy())
            result_dict['pca_2'][this_index][1].extend(projected_data[:, -2].cpu().detach().numpy())
            
            result_dict['pca_3'][this_index][0].extend(projected_data[:, -1].cpu().detach().numpy())
            result_dict['pca_3'][this_index][1].extend(projected_data[:, -2].cpu().detach().numpy())
            result_dict['pca_3'][this_index][2].extend(projected_data[:, -3].cpu().detach().numpy())

            random_dims  = random.sample(set(list(range(0, data.shape[1]))), 2)
            result_dict['random_2'][this_index][0].extend(data[0:data.shape[0], random_dims[0]].cpu().detach().numpy())
            result_dict['random_2'][this_index][1].extend(data[0:data.shape[0], random_dims[1]].cpu().detach().numpy())

            random_dims  = random.sample(set(list(range(0, data.shape[1]))), 3)
            result_dict['random_3'][this_index][0].extend(data[0:data.shape[0], random_dims[0]].cpu().detach().numpy())
            result_dict['random_3'][this_index][1].extend(data[0:data.shape[0], random_dims[1]].cpu().detach().numpy())
            result_dict['random_3'][this_index][2].extend(data[0:data.shape[0], random_dims[2]].cpu().detach().numpy())
        elif data.shape[1] == 3:

            result_dict['pca_2'][this_index][0].extend(projected_data[:, -1].cpu().detach().numpy())
            result_dict['pca_2'][this_index][1].extend(projected_data[:, -2].cpu().detach().numpy())

            random_dims  = random.sample(set(list(range(0, data.shape[1]))), 2)
            result_dict['random_2'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['random_2'][this_index][1].extend(data[:, 1].cpu().detach().numpy())

            result_dict['pca_3'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['pca_3'][this_index][1].extend(data[:, 1].cpu().detach().numpy())
            result_dict['pca_3'][this_index][2].extend(data[:, 2].cpu().detach().numpy())

            result_dict['random_3'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['random_3'][this_index][1].extend(data[:, 1].cpu().detach().numpy())
            result_dict['random_3'][this_index][2].extend(data[:, 2].cpu().detach().numpy())
            
        elif data.shape[1] == 2:
            result_dict['pca_2'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['pca_2'][this_index][1].extend(data[:, 1].cpu().detach().numpy())

            result_dict['random_2'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['random_2'][this_index][1].extend(data[:, 1].cpu().detach().numpy())

            result_dict['pca_3'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['pca_3'][this_index][1].extend(data[:, 1].cpu().detach().numpy())
            result_dict['pca_3'][this_index][2].extend(data[:, 1].cpu().detach().numpy() * 0)

            result_dict['random_3'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['random_3'][this_index][1].extend(data[:, 1].cpu().detach().numpy())
            result_dict['random_3'][this_index][2].extend(data[:, 1].cpu().detach().numpy() * 0)
        else:
            result_dict['pca_2'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['pca_2'][this_index][1].extend(data[:, 0].cpu().detach().numpy() * 0)

            result_dict['random_2'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['random_2'][this_index][1].extend(data[:, 0].cpu().detach().numpy() * 0)

            result_dict['pca_3'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['pca_3'][this_index][1].extend(data[:, 0].cpu().detach().numpy() * 0)
            result_dict['pca_3'][this_index][2].extend(data[:, 0].cpu().detach().numpy() * 0)

            result_dict['random_3'][this_index][0].extend(data[:, 0].cpu().detach().numpy())
            result_dict['random_3'][this_index][1].extend(data[:, 0].cpu().detach().numpy())
            result_dict['random_3'][this_index][2].extend(data[:, 0].cpu().detach().numpy() * 0)

        return result_dict

    # covar_matrix = covariance_matrix.cpu().detach().numpy()

    values, vectors = torch.linalg.eigh(covariance_matrix)

    projected_data = torch.matmul(data, vectors)

    if first_batch:
        result_dict['pca_2'].append(([], []))
        result_dict['pca_3'].append(([], [], []))
        result_dict['random_2'].append(([], []))
        result_dict['random_3'].append(([], [], []))
        result_dict['norms'].append([])
        result_dict['spherical_mean_width_v2'].append(0)

    # if preds is not None:
    #     result_dict['labels'].extend(preds)
        
    result_dict['spherical_mean_width_v2'][this_index] = (result_dict['spherical_mean_width_v2'][this_index] + calc_spherical_mean_width_v2(data).detach().cpu().numpy()) / 2.0

    result_dict['norms'][this_index].extend(distance_from_origin(data).detach().cpu().numpy().tolist())

    result_dict = get_projected_data(result_dict, projected_data, data)

    del values, vectors
    gc.collect()


    return result_dict


def create_gif_from_plots(plot_dir, output_gif, plot_type, start_epoch=10, end_epoch=100, step=10, gif_duration=1500):
    frames = []
    font = ImageFont.load_default()  # You can replace this with a path to a font file if you need a custom font

    for epoch in range(start_epoch, end_epoch + 1, step):
        # Load both train and validation images for each epoch
        # img_path = os.path.join(plot_dir, f'epoch_{epoch}/_{plot_type}_all_plots.pdf')
        img_path = os.path.join(plot_dir, f'epoch_{epoch}/_{plot_type}_all_plots.png')

        # Open the image (train or validation plot)
        img = Image.open(img_path)

        # Add epoch title to the plot
        draw = ImageDraw.Draw(img)
        text = f"Epoch {epoch} - {plot_type.capitalize()}"
        text_width, text_height = draw.textsize(text, font=font) # type: ignore
        draw.text(((img.width - text_width) // 2, 10), text, font=font, fill="white")

        # Append the frame to the list
        frames.append(img)

    # Save the frames as a GIF
    frames[0].save(output_gif, format='GIF', append_images=frames[1:], save_all=True, duration=gif_duration, loop=0)



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

