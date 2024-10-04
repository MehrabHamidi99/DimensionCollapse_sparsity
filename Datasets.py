from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import torch


# Custom dataset class to remap the labels into three classes
class ThreeClassMNIST(Dataset):
    def __init__(self, original_dataset, class_mapping):
        self.dataset = original_dataset
        self.class_mapping = class_mapping
        
        # Create a mapping function based on the provided class_mapping
        self.label_map = {}
        for i, cls in enumerate(class_mapping):
            for label in cls:
                self.label_map[label] = i

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Remap the label into one of the three classes
        new_label = self.label_map[label]
        return img, new_label
    


class CustomCIFAR10Dataset(Dataset):
    def __init__(self, cifar10_dataset):
        self.data = cifar10_dataset.data # Original shape (50000, 32, 32, 3)
        self.targets = cifar10_dataset.targets
        
        # Convert data to a PyTorch tensor and change the shape
        self.data = torch.Tensor(self.data.transpose(0, 3, 1, 2))  # Change shape to (50000, 3, 32, 32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]
