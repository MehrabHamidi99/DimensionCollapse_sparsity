import os

from Experiments import *
from tqdm import tqdm
from Models import *
from utils import *
from DataGenerator import *
from DataLoader import *

import time

from torchvision import datasets, transforms



import torch
import torch.nn.functional as F

# # Example small batch with fixed labels 0 0 1 1 2 2 0 0 1 1 2 2
# fixed_labels = torch.tensor([0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2])

# # Generate random samples for each label (in this case, we use MNIST image size, e.g., 1x28x28)
# fixed_samples = torch.randn((len(fixed_labels), 1, 28, 28))  # Adjust size as needed for your model

# # Send to device if needed (assuming GPU usage)
# fixed_samples = fixed_samples
# fixed_labels = fixed_labels

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

arch = (784, [128, 64, 3])

# model = MNIST_classifier(n_in=arch[0], layer_list=arch[1], bias=0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# feature_extractor = ReluExtractor(model, device=device)


# # Forward pass through the model
# output = model(fixed_samples)
# preds = torch.argmax(F.softmax(output, dim=1), dim=1)

# # Register activations via hooks, or use pre-registered hooks to collect activations
# relu_outputs = hook_forward(feature_extractor, fixed_samples, fixed_labels, device)

# # Collect activations for each layer
# for i, activations in enumerate(relu_outputs):
#     print(f"Layer {i + 1} activations shape: {activations.shape}")

# # Print out the labels and check if activations correspond to the expected pattern
# print("Fixed Labels: ", fixed_labels.cpu().numpy())
# print("Predicted Labels: ", preds.cpu().numpy())

# # Optionally, print the first few activations of the batch to verify correspondence
# for i, activations in enumerate(relu_outputs):
#     print(f"Layer {i + 1} activations for first few samples: \n{activations[:3].cpu().numpy()}")



mnist_training_analysis_hook_engine(0, archirecture=arch, pre_path='debug_res', three_class=True, odd_even=False)


# mnist_training_analysis_hook_engine(0, archirecture=(784, [256, 128, 64, 32, 10]), pre_path='sept_result_mnist_three_class', three_class=True)
# # train_iter, test_iter = IMDB()

# # # Pre-trained model
# # model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

# mnist_training_analysis_hook_engine(0, pre_path='debug_here')

# scale = 10
# loc = 0
# normal_dist = True
# archs = [
#     # (2, [10, 20, 30]),
#     # (2, [2, 2, 2]),
#     # (2, [2, 2, 2, 2]),
#     # (2, [2, 2, 2, 2, 2]),
#     # (3, [3, 3, 3, 3, 3]),

    
#     (100, [100 for _ in range(10)]),
#     # (2, [16, 64, 256, 256, 256, 256]),
#     # (100, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
#     # (2, [4])
#     # (2, [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
# ]


# data_properties = {
#     'exp_type':'fixed',
#     'normal_dist': normal_dist, 
#     'loc': loc, 
#     'scale': scale
# }


# s_t = time.time()
# # results = [random_experiment_hook_engine(i, exps=1, num=10000, pre_path='test_results/', data_properties=data_properties, model_type='mlp') for i in tqdm(archs)]
# results = [batch_fixed_model_hook_engine(i, None, data_properties, 50000) for i in tqdm(archs)]

# # # print(time.time() - s_t)