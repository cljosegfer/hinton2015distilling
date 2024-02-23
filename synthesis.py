# setup
import torch

from PIL import Image
from torchvision import transforms

from datasets import load_dataset

from tqdm import tqdm
import os

BATCH_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

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

# data
trn = load_dataset("evanarlian/imagenet_1k_resized_256", split = 'train')

trn = trn.with_transform(transform)
trn_loader = torch.utils.data.DataLoader(trn, collate_fn = collate_fn, batch_size = BATCH_SIZE)

# model
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
# model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)

model = model.to(device)

backbone = list(model.children())[:-1]
model = torch.nn.Sequential(*backbone)

# synthesis
H_trn = torch.empty(size = [0])

model.eval()
with torch.no_grad():
    for i, sample in tqdm(enumerate(trn_loader)):
        x = sample['image'].to(device)
        h = model.forward(x)

        H_trn = torch.cat((H_trn, h.cpu()))

# write
if not os.path.exists('output'):
    os.makedirs('output')

torch.save(H_trn, 'output/H_train.pt')
# H = torch.load('output/H_trn.pt')
