# setup
import torch

from PIL import Image
from torchvision import transforms

from datasets import load_dataset
from utils import transform, collate_fn, model_select

from tqdm import tqdm
import os

BATCH_SIZE = 256

device = "cuda" if torch.cuda.is_available() else "cpu"

# data
trn = load_dataset("evanarlian/imagenet_1k_resized_256", split = 'train')
val = load_dataset("evanarlian/imagenet_1k_resized_256", split = 'val')

trn = trn.with_transform(transform)
trn_loader = torch.utils.data.DataLoader(trn, collate_fn = collate_fn, batch_size = BATCH_SIZE)
val = val.with_transform(transform)
val_loader = torch.utils.data.DataLoader(val, collate_fn = collate_fn, batch_size = BATCH_SIZE)

# model
model_label = 'rn18'
model = model_select(model_label)

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

H_val = torch.empty(size = [0])

model.eval()
with torch.no_grad():
    for i, sample in tqdm(enumerate(val_loader)):
        x = sample['image'].to(device)
        h = model.forward(x)

        H_val = torch.cat((H_val, h.cpu()))

# write
if not os.path.exists('output'):
    os.makedirs('output')

torch.save(H_trn, 'output/{}_H_train.pt'.format(model_label))
torch.save(H_val, 'output/{}_H_val.pt'.format(model_label))
