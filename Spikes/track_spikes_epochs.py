import os
import sys
from tqdm import tqdm
import numpy as np
import torch
import matplotlib.pyplot as plt

import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/../')))


from Data.DataLoader import *
from Models.Models_normal import MNIST_classifier
from Training.Analysis import fixed_model_batch_analysis
from Training.Spike_loss import *


def load_model(epoch, model_path_template, arch, device):
    """
    Load the model for a specific epoch.
    """
    model = MNIST_classifier(n_in=arch[0], layer_list=arch[1], bias=0)
    model.to(device)
    model_path = model_path_template.format(epoch)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def perform_spike_detection(model, dataset_samples, dataset_labels, device, layer_idx, anal_path):
    """
    Perform spike detection for a specific layer and return spike assignments.
    """
    results_dict = fixed_model_batch_analysis(
        model, dataset_samples, dataset_labels, device, f'{anal_path}_layer_{layer_idx}_', 'analyze', plotting=False
    )
    
    mnist_labels = dataset_labels.detach().cpu().numpy()
    mnist_pca_2d = np.array(results_dict['pca_2'][layer_idx]).transpose()
    
    num_classes = len(np.unique(mnist_labels))
    
    spike_info_per_class = {}
    class_idx = 0
    
    for class_idx in tqdm(range(num_classes), desc=f"Processing Class {class_idx}"):
        # Extract class data
        class_data_indices = np.where(mnist_labels == class_idx)[0]
        points = np.array(mnist_pca_2d[class_data_indices])
        
        # Detect spikes (hyperplanes)
        detected_hyperplanes, total_error, assigned_points = spike_detection_nd(points)
        
        # Assign points to hyperplanes
        assignments, _, _ = assign_points_to_hyperplanes(torch.Tensor(points), detected_hyperplanes)
        
        spike_info_per_class[class_idx] = {
            'detected_hyperplanes': detected_hyperplanes,
            'assignments': assignments,
            'class_data_indices': class_data_indices
        }
    
    return spike_info_per_class, results_dict

def track_spikes_across_epochs(model_path_template, arch, device, dataset_samples, dataset_labels, layer_idx, epochs, output_dir):
    """
    Track spikes for a specific layer across multiple epochs.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store spike information per epoch and per class
    spikes_across_epochs = {epoch: {} for epoch in epochs}
    
    for epoch in tqdm(epochs, desc="Processing Epochs"):
        # Load model for the current epoch
        model = load_model(epoch, model_path_template, arch, device)
        
        # Define analysis path for current epoch
        anal_path = os.path.join(output_dir, f'epoch_{epoch}')
        os.makedirs(anal_path, exist_ok=True)
        
        # Perform spike detection
        spike_info_per_class, results_dict = perform_spike_detection(
            model, dataset_samples, dataset_labels, device, layer_idx, anal_path
        )
        
        # Store spike assignments
        spikes_across_epochs[epoch] = spike_info_per_class
    
    return spikes_across_epochs

def plot_spike_tracking(spikes_across_epochs, epochs, class_idx, layer_idx, output_dir):
    """
    Plot the tracking of spikes for a specific class and layer across epochs.
    """
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(epochs)))
    
    for color, epoch in zip(colors, epochs):
        spike_info = spikes_across_epochs[epoch][class_idx]
        detected_hyperplanes = spike_info['detected_hyperplanes']
        
        # Assuming hyperplanes can be represented as points (e.g., cluster centers)
        # Modify this based on how your hyperplanes are represented
        for spike_idx, hyperplane in enumerate(detected_hyperplanes):
            plt.scatter(
                hyperplane[0], hyperplane[1],
                color=color, label=f'Epoch {epoch}' if spike_idx == 0 else "",
                marker='o', edgecolors='k', alpha=0.7
            )
    
    plt.title(f'Spike Tracking for Class {class_idx} in Layer {layer_idx}')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend(title='Epochs', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, f'spike_tracking_class_{class_idx}_layer_{layer_idx}.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f'Spike tracking plot saved to {save_path}')

def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arch = (784, [256, 128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10])
    
    # Path template for model checkpoints
    # Example: '/path/to/models/epoch_{epoch}/model.pt'
    model_path_template = '/home/mila/m/mehrab.hamidi/scratch/training_res/january_res/mnist/normal/bias_0.0001/mnist_training/try_num3/epoch_{}/model.pt'
    
    # Load the data
    _, _, _, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = get_mnist_data_loaders()
    
    dataset_target_samples = train_samples
    dataset_target_labels = train_labels
    
    # Define epochs to track
    epochs = [1, 2, 3, 10, 20, 50, 60, 100, 200]
    
    # Specify the layer to track
    layer_idx = 7  # Change as needed
    
    # Output directory for analysis using absolute path in user's home
    output_dir = os.path.expanduser('~/spike_analysis/spike_tracking_across_epochs/')
    os.makedirs(output_dir, exist_ok=True)
    
    # Track spikes across epochs
    spikes_across_epochs = track_spikes_across_epochs(
        model_path_template, arch, device,
        dataset_target_samples, dataset_target_labels,
        layer_idx, epochs, output_dir
    )
    
    # Plot spike tracking for each class
    num_classes = len(np.unique(train_labels.detach().cpu().numpy()))
    
    for class_idx in range(num_classes):
        plot_spike_tracking(
            spikes_across_epochs, epochs, class_idx,
            layer_idx, output_dir
        )

if __name__ == "__main__":
    main()
