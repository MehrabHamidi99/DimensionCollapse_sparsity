from utils import *
from DataGenerator import *
from Models_normal import *
import torch.optim as optim
from Analysis import fixed_model_batch_analysis

from sklearn.neighbors import KernelDensity
from sklearn.mixture import GaussianMixture

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = 'bayesian_est1/'
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
    
    plt.contourf(xx, yy, decision_boundary, levels=50, cmap='coolwarm', alpha=0.5)
    
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap='bwr', alpha=0.6)
    
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.title("Decision Boundary with Gaussian Mixture Models")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.savefig(save_path + 'decision_boundary.png')


def sample_gaussian_data(input_dimension=2, num=1000):
    mean1 = np.ones(input_dimension) * 2        # Mean of the first class (positive label)
    variance1 = 0.5                            # Variance of the first class
    data1 = create_random_data_normal_dist(input_dimension=input_dimension, num=num, loc=mean1, scale=np.sqrt(variance1))
    
    mean2 = np.ones(input_dimension) * -25     
    variance2 = 100                            
    data2 = create_random_data_normal_dist(input_dimension=input_dimension, num=num, loc=mean2, scale=np.sqrt(variance2))
    
    X = np.vstack((data1, data2))
    y = np.array([1] * num + [0] * num)  
    
    return X, y


save_path = 'bayesian_est1/'
if not os.path.isdir(save_path):
    os.makedirs(save_path)
    
######## Main execution
d_ = 5  
try_num = 4

save_path += '{}/{}/'.format(str(5), str(try_num))
if not os.path.isdir(save_path):
    os.makedirs(save_path)


X, y = sample_gaussian_data(d_, 10000)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# # Fit Gaussian Mixture Models for both classes
# gmm1 = GaussianMixture(n_components=1).fit(X[y == 1])
# gmm2 = GaussianMixture(n_components=1).fit(X[y == 0])


# xx, yy, decision_boundary = estimate_decision_boundary(X, gmm1, gmm2)
# plot_decision_boundary(X, y, xx, yy, decision_boundary)


# X = torch.tensor(X, dtype=torch.float32).to(device)
# y = torch.tensor(y, dtype=torch.float32)
X = X.to(device)
y = y.to(device)

input_dimension = X.shape[1]  
# layer_list = [16, 8, 1]       # try num 1 100%
# layer_list = [16, 16, 16, 16, 16, 8, 1]      # try num 2  100%

layer_list = [16, 16, 16, 16, 16, 8, 8, 8, 8, 1]      # try num 3 50%
layer_list = [16, 16, 8, 8, 1]      # try num 4 50%

model = MLP_simple(input_dimension, layer_list).to(device)

criterion = nn.BCEWithLogitsLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001) # type: ignore

num_epochs = 500

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()  
    outputs = model(X).squeeze()  
    loss = criterion(outputs, y) 
    loss.backward()
    optimizer.step()


    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            predictions = torch.sigmoid(outputs).squeeze() 
            predicted_labels = (predictions > 0.5).float()
            
            train_accuracy = (predicted_labels == y).float().mean()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Train Accuracy: {train_accuracy.item() * 100:.2f}%')


model.eval()
with torch.no_grad():
    predictions = torch.sigmoid(model(X)).squeeze() 
    predicted_labels = (predictions > 0.5).float() 

X = X.detach().cpu()
predicted_labels = predicted_labels.detach().cpu()

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