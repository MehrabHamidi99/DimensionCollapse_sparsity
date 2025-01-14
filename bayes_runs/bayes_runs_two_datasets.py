
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/../.')))

from utils import *
from Data.DataGenerator import *
from Models.Models_normal import *
import torch.optim as optim
from Training.Analysis import fixed_model_batch_analysis

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from data_classes_clusters import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = 'bayesian_est_good2/'
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


def plot_decision_boundary(X, y, xx, yy, decision_boundary, name=None):
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


X, y = generate_data(mode='4island')

# Fit Gaussian Mixture Models for both classes
gmm1 = GaussianMixture(n_components=1).fit(X[y == 1])
gmm2 = GaussianMixture(n_components=1).fit(X[y == 0])


xx, yy, decision_boundary = estimate_decision_boundary(X, gmm1, gmm2)
plot_decision_boundary(X, y, xx, yy, decision_boundary)


X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.long)
X = X.to(device)
y = y.to(device)

# 2. Initialize the MLP
input_dimension = X.shape[1]  # Should be 2
# layer_list = [16, 16, 16, 16, 8, 8, 8, 4, 2]        # Hidden layers configuration  ---- good
layer_list = [16, 16, 16, 8, 8, 8, 8, 8, 4, 2]        # Hidden layers configuration  ---- good1
layer_list = [16, 16, 16, 8, 8, 8, 8, 4, 2]        # Hidden layers configuration  ---- good2 removing layer 6!

model = MLP_simple(input_dimension, layer_list).to(device)

# 3. Define loss function and optimizer
# criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001) # type: ignore

# 4. Training loop
num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Clear gradients
    outputs = model(X) # Forward pass
    loss = criterion(outputs, y)  # Calculate loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights


    if (epoch + 1) % 10 == 0:
        # Calculate training accuracy
        with torch.no_grad():
            # predictions = torch.sigmoid(outputs).squeeze()  # Apply sigmoid to get probabilities
            # predicted_labels = (predictions > 0.5).float()  # Convert probabilities to binary labels
            
            # train_accuracy = (predicted_labels == y).float().mean()  # Calculate accuracy

            predictions = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            predicted_labels = torch.argmax(predictions, dim=1).float()  # Convert probabilities to labels
            train_accuracy = (predicted_labels == torch.tensor(y, dtype=torch.float32).to('cuda')).float().mean()  # Calculate accuracy

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy.item() * 100:.2f}%')

# 5. Evaluate the model
model.eval()
with torch.no_grad():
    # predictions = torch.sigmoid(model(X)).squeeze()  # Apply sigmoid to get probabilities
    # predicted_labels = (predictions > 0.5).float()    # Convert probabilities to binary labels

    predictions = torch.softmax(model(X), dim=1)[:, 1]  # Apply softmax to get probabilities for class 1
    predicted_labels = (predictions > 0.5).float()  # Convert probabilities to binary labels

X = X.detach().cpu()
predicted_labels = predicted_labels.detach().cpu()


xx, yy, decision_boundary = estimate_decision_boundary_pytorch(X, model)
plot_decision_boundary(X.numpy(), y.detach().cpu().numpy(), xx, yy, decision_boundary, 'mlp_decision_boundary')



# Visualization of the results
plt.figure(figsize=(10, 8))
plt.scatter(X[predicted_labels == 1][:, 0], X[predicted_labels == 1][:, 1], color='blue', label='Class 1 (Positive)', alpha=0.5)
plt.scatter(X[predicted_labels == 0][:, 0], X[predicted_labels == 0][:, 1], color='red', label='Class 0 (Negative)', alpha=0.5)
plt.title("MLP Predictions on Gaussian Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid()
plt.savefig(save_path + 'mlp_predictions.png')


fixed_model_batch_analysis(model=model , samples=X, labels=y, device=device, save_path=save_path, model_status='')