from utils import *
from DataGenerator import *
from Models_normal import *
import torch.optim as optim
from Analysis import fixed_model_batch_analysis
from DataLoader import *

import os

print(os.getcwd())
# os.chdir()

# Load the model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MNIST_classifier(n_in=784, layer_list=[128, 128, 128, 64, 64, 64, 64, 64, 64, 32, 10], bias=0)
state_dict = torch.load('running_scripts/sept_result_mnist/mnist_training/try_num0/epoch_30/model.pth', weights_only=False)
model.load_state_dict(state_dict)

model.eval()


# load the data
_, _, _, train_samples, train_labels, val_samples, val_labels, test_samples, test_labels = get_mnist_data_loaders()

anal_path = 'spike_analysis/mnist/'
if not os.path.isdir(anal_path):
    os.makedirs(anal_path)
results_dict = fixed_model_batch_analysis(model, train_samples, train_labels, device, '{}_{}'.format('spike_analysis/mnist/', '_train'), 'analyze')

