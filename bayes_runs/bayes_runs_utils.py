from utils import *
from DataGenerator import *
from Models_normal import *
import torch.optim as optim
from Analysis import fixed_model_batch_analysis

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from data_classes_clusters import *

import os
import numpy as np

def plot_loss_and_accuracy(losses, accuracies, save_path):
    # Plot loss over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + 'loss_plot.png')

    # Plot accuracy over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path + 'accuracy_plot.png')


save_path = 'universal_approximation_donut/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)

def estimate_decision_boundary(X, gmm1, gmm2):
    # Generate a grid to evaluate the density
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # Compute the density for both classes
    log_density1 = gmm1.score_samples(grid)
    log_density2 = gmm2.score_samples(grid)

    # Calculate the decision boundary
    decision_boundary = np.exp(log_density1) - np.exp(log_density2)
    
    return xx, yy, decision_boundary.reshape(xx.shape)

def estimate_decision_boundary_pytorch(X, mlp_model, device='cuda'):
    # Generate a grid to evaluate the density
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # Convert the grid to a PyTorch tensor and move it to the specified device
    grid_tensor = torch.tensor(grid, dtype=torch.float32).to(device)

    # Set the model to evaluation mode
    mlp_model.eval()
    
    # Disable gradient calculation for efficiency
    with torch.no_grad():
        # Pass the grid through the model
        outputs = mlp_model(grid_tensor)
        
        # Apply sigmoid to get probabilities
        # probabilities = torch.sigmoid(outputs).squeeze()

        # Apply softmax to get probabilities
        probabilities = torch.softmax(outputs, dim=1)[:, 1]

        # Get the probability difference between the two classes
        decision_boundary = probabilities - (1 - probabilities)
    
    # Move the decision boundary back to CPU and convert to numpy
    decision_boundary = decision_boundary.cpu().numpy()
    
    return xx, yy, decision_boundary.reshape(xx.shape)

def plot_decision_boundary(X, y, xx, yy, decision_boundary, save_path, name=None):
    plt.figure(figsize=(10, 8))
    
    # Plot the decision boundary
    plt.contourf(xx, yy, decision_boundary, levels=50, cmap='coolwarm', alpha=0.5)
    
    # Plot data points
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='bwr', alpha=0.6)
    
    # Add a legend
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("Decision Boundary with Gaussian Mixture Models")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    if name is None:
        plt.savefig(save_path + 'decision_boundary.png')
    else:
        plt.savefig(save_path + name)



# Training loop with loss and accuracy tracking
def train_model(model, X, y, criterion, optimizer, scheduler, num_epochs, save_path):
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()  # Clear gradients
        outputs = model(X)  # Forward pass
        loss = criterion(outputs, y)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights
        scheduler.step()  # Step the scheduler

        if (epoch + 1) % 10 == 0:
            # Calculate training accuracy
            with torch.no_grad():
                predictions = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
                predicted_labels = torch.argmax(predictions, dim=1).float()  # Convert probabilities to labels
                train_accuracy = (predicted_labels == torch.tensor(y, dtype=torch.float32).to('cuda')).float().mean()  # Calculate accuracy

            losses.append(loss.item())
            accuracies.append(train_accuracy.item() * 100)

            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy.item() * 100:.2f}%')
            
            # Save accuracy, loss, and model configuration to file
            with open(save_path + 'training_log.txt', 'a') as f:
                f.write(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy.item() * 100:.2f}%\n')

    # Plot and save loss and accuracy
    plot_loss_and_accuracy(losses, accuracies, save_path)
    return model