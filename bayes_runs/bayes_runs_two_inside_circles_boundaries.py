import os, sys; sys.path.append(os.path.dirname(os.path.realpath(f'{__file__}/../.')))

from utils import *
from Data.DataGenerator import *
from Models.Models_normal import *
import torch.optim as optim
from Training.Analysis import fixed_model_batch_analysis

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

from data_classes_clusters import *

import os
import numpy as np

from bayes_runs_utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

layer_list = [8, 8, 8, 8, 8, 4, 2]        # Hidden layers configuration  ---- good
# layer_list = [16, 16, 16, 8, 8, 8, 8, 8, 4, 2]        # Hidden layers configuration  ---- good1
# layer_list = [16, 16, 16, 8, 8, 8, 8, 4, 2]        # Hidden layers configuration  ---- good2 removing layer 6!


# 4. Training loop
num_epochs = 5000

try_num = 6 # Define try number here

save_path = 'donuts_and_islands/'

save_path = f'{save_path}try_{try_num}/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)


X, y = generate_data(mode='donuts_and_islands') # type: ignore

print(X.shape)
print(y.shape)

# Fit Gaussian Mixture Models for both classes
gmm1 = GaussianMixture(n_components=1).fit(X[y == 1])
gmm2 = GaussianMixture(n_components=1).fit(X[y == 0])

xx, yy, decision_boundary = estimate_decision_boundary(X, gmm1, gmm2)
plot_decision_boundary(X, y, xx, yy, decision_boundary, save_path)

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.long)
X = X.to(device)
y = y.to(device)

# 2. Initialize the MLP
input_dimension = X.shape[1]  # Should be 2

model = MLP_simple(input_dimension, layer_list).to(device)

# 3. Define loss function and optimizer
# criterion = nn.BCEWithLogitsLoss()  # Binary Cross-Entropy with logits
criterion = nn.CrossEntropyLoss()  # Cross-Entropy Loss for multi-class classification
optimizer = optim.Adam(model.parameters(), lr=0.001) # type: ignore
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.1)


train_model(model, X, y, criterion, optimizer, scheduler, num_epochs, save_path)


# Save model configuration
with open(save_path + 'model_config.txt', 'w') as f:
    model_architecture = str(model)
    f.write(f"Model full arch: {model_architecture}\n")
    f.write(f'Model Architecture: {layer_list}\n')
    f.write(f'Data Configuration: mode=inside_circles, num_samples={X.shape[0]}\n')
    f.write(f"number of epoch {num_epochs}\n")

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
plot_decision_boundary(X.numpy(), y.detach().cpu().numpy(), xx, yy, decision_boundary, save_path, name='mlp_decision_boundary')

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
