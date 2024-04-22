
import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import Dataset

TRANSFORM = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

class Cifar10(Dataset):
    def __init__(self, split, embedding_path = None):
        self.transform = TRANSFORM[split]
        self.data = torchvision.datasets.CIFAR10(root='/home/josegfer/datasets/cifar', 
                                                 train = (split == 'train'), transform = self.transform)
        self.embedding = False
        if embedding_path is not None:
            self.H = torch.load(embedding_path)
            self.embedding = True
    
    def __len__(self):
        return self.data.__len__()
    
    def __getitem__(self, idx):
        x = self.data.__getitem__(idx)[0]
        y = self.data.__getitem__(idx)[1]
        if self.embedding:
            return {'image': x, 'label': y, 'embedding': self.H[idx, :, 0, 0]}
        return {'image': x, 'label': y}
