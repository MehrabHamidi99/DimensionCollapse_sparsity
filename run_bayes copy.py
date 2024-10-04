from utils import *
from DataGenerator import *
from Models_normal import *
import torch.optim as optim
from Analysis import fixed_model_batch_analysis

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = 'bayesian_est/'
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


def plot_decision_boundary(X, y, xx, yy, decision_boundary):
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
    plt.savefig(save_path + 'decision_boundary.png')

mean1 = [2, 2]        # Mean of the first class (positive label)
variance1 = 0.5      # Variance of the first class
data1 = create_random_data_normal_dist(input_dimension=2, num=1000, loc=mean1, scale=np.sqrt(variance1))

# Parameters for the second Gaussian distribution
mean2 = [-25, -25]     # Mean of the second class (negative label)
variance2 = 100       # Variance of the second class (much larger)
data2 = create_random_data_normal_dist(input_dimension=2, num=1000, loc=mean2, scale=np.sqrt(variance2))

# Combine the datasets and create labels
X = np.vstack((data1, data2))
y = np.array([1] * 1000 + [0] * 1000)  # Positive labels for the first class and negative for the second


# Fit Gaussian Mixture Models for both classes
gmm1 = GaussianMixture(n_components=1).fit(X[y == 1])
gmm2 = GaussianMixture(n_components=1).fit(X[y == 0])


xx, yy, decision_boundary = estimate_decision_boundary(X, gmm1, gmm2)
plot_decision_boundary(X, y, xx, yy, decision_boundary)


X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32)
X = X.to(device)
y = y.to(device)

# 2. Initialize the MLP
input_dimension = X.shape[1]  # Should be 2
layer_list = [16, 8, 1]        # Hidden layers configuration
model = MLP_simple(input_dimension, layer_list).to(device)

# 3. Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
optimizer = optim.Adam(model.parameters(), lr=0.001) # type: ignore

# 4. Training loop
num_epochs = 200

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  # Clear gradients
    outputs = model(X).squeeze()  # Forward pass
    loss = criterion(outputs, y)  # Calculate loss
    loss.backward()  # Backward pass
    optimizer.step()  # Update weights


    if (epoch + 1) % 10 == 0:
        # Calculate training accuracy
        with torch.no_grad():
            predictions = torch.sigmoid(outputs).squeeze()  # Apply sigmoid to get probabilities
            predicted_labels = (predictions > 0.5).float()  # Convert probabilities to binary labels
            
            train_accuracy = (predicted_labels == y).float().mean()  # Calculate accuracy

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy.item() * 100:.2f}%')

# 5. Evaluate the model
model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(X)).squeeze()  # Apply sigmoid to get probabilities
    predicted_labels = (predictions > 0.5).float()    # Convert probabilities to binary labels

X = X.detach().cpu()
predicted_labels = predicted_labels.detach().cpu()

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


fixed_model_batch_analysis(model=model , samples=X, labels=y, device=device, save_path=save_path, model_status='2, 16, 8, 1')