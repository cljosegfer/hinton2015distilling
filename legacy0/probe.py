
# setup
import torch

from PIL import Image
from torchvision import transforms

from datasets import load_dataset
from utils import transform, collate_fn, Imagenet_h

from tqdm import tqdm

BATCH_SIZE = 256
EPOCHS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

# data
trn = load_dataset("evanarlian/imagenet_1k_resized_256", split = 'train')
val = load_dataset("evanarlian/imagenet_1k_resized_256", split = 'val')

trn = trn.with_transform(transform)
val = val.with_transform(transform)

H_trn = torch.load('output/rn34_H_train.pt')
H_val = torch.load('output/rn34_H_val.pt')

trn = Imagenet_h(trn, H_trn)
val = Imagenet_h(val, H_val)

trn_loader = torch.utils.data.DataLoader(trn, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE)

# model
linear_probe = torch.nn.Sequential(
    torch.nn.Linear(in_features = 512, out_features = 1000, bias = True),
    torch.nn.Softmax(),
)
linear_probe = linear_probe.to(device)

# train
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(linear_probe.parameters())

linear_probe.train()
for epoch in (range(EPOCHS)):
    for x, y in tqdm(trn_loader):
        x = x.to(device)
        y = y.to(device)

        output = linear_probe.forward(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# eval
total = 0
correct = 0

linear_probe.eval()
with torch.no_grad():
    for x, y in tqdm(val_loader):
        x = x.to(device)
        y = y.to(device)

        output = linear_probe.forward(x)
        _, predicted = torch.max(output.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
print(1 - correct / total)
