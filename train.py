
# setup

import torch
import os

from utils import H_imagenet, train, eval, model_select

from tqdm import tqdm

BATCH_SIZE = 128
EPOCHS = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

# data

trn = H_imagenet('train', 'rn34')
val = H_imagenet('val', 'rn34')

trn_loader = torch.utils.data.DataLoader(trn, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE)

# model
model_label = 'rn18'
model = model_select(model_label)

model = model.to(device)

backbone = list(model.children())[:-1]
model = torch.nn.Sequential(*backbone)

# train
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

log = []
for epoch in (range(EPOCHS)):
    loss_trn = train(model, trn_loader, optimizer, criterion, device)
    loss_val = eval(model, val_loader, criterion, device)
    log.append([loss_trn, loss_val])

print(log)

# write
if not os.path.exists('output'):
    os.makedirs('output')

torch.save(model.cpu(), 'output/rn18_mirror_rn34.pt'.format(model_label))
