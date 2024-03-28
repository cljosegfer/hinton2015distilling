
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from datasets import load_dataset

from tqdm import tqdm

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def transform(examples):
    examples['image'] = [preprocess(image.convert("RGB")) for image in examples['image']]
    return examples

def collate_fn(examples):
    images = []
    labels = []
    for example in examples:
        images.append((example['image']))
        labels.append(example['label'])

    images = torch.stack(images)
    labels = torch.tensor(labels)
    return {'image': images, 'label': labels}

class H_imagenet(Dataset):
    def __init__(self, split, model_label):
        self.imagenet = load_dataset("evanarlian/imagenet_1k_resized_256", split = split)
        self.imagenet = self.imagenet.with_transform(transform)
        self.H = torch.load('output/{}_H_{}.pt'.format(model_label, split))
    
    def __len__(self):
        return self.imagenet.__len__()
    
    def __getitem__(self, idx):
        x = self.imagenet.__getitem__(idx)['image']
        h = self.H[idx, :, :, :]

        return x, h

class Imagenet_h(Dataset):
    def __init__(self, imagenet, H):
        self.imagenet = imagenet
        self.imagenet = self.imagenet.with_transform(transform)
        self.H = H
    
    def __len__(self):
        return self.imagenet.__len__()
    
    def __getitem__(self, idx):
        y = self.imagenet.__getitem__(idx)['label']
        h = self.H[idx, :, 0, 0]

        return h, y

def train(model, loader, optimizer, criterion, device):
    model.train()
    custo = 0
    for x, h in tqdm(loader):
        x = x.to(device)
        h = h.to(device)

        output = model.forward(x)
        loss = criterion(output, h)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        custo += loss.item()
    
    return custo / len(loader)

def eval(model, loader, criterion, device, synthesis = False):
    model.eval()
    custo = 0
    with torch.no_grad():
        for x, h in loader:
            x = x.to(device)
            h = h.to(device)

            output = model.forward(x)
            loss = criterion(output, h)

            custo += loss.item()
    if synthesis:
        return custo / len(loader), h, output
    return custo / len(loader)

def model_select(model_label):
    if model_label == 'rn18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained = True)
    if model_label == 'rn34':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained = True)
    if model_label == 'rn50':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained = True)
    if model_label == 'rn101':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained = True)
    if model_label == 'rn152':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained = True)
    
    return model
