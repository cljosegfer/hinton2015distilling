
# setup

import torch
import os

from utils import H_imagenet, train, eval, model_select, plot_log

from tqdm import tqdm

BATCH_SIZE = 128
EPOCHS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists('output'):
    os.makedirs('output')

# data
origem = 'rn18'
espelhado = 'rn34'
model_label = origem + espelhado

trn = H_imagenet('train', espelhado)
val = H_imagenet('val', espelhado)

trn_loader = torch.utils.data.DataLoader(trn, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE)

# model
model = model_select(origem)

model = model.to(device)

backbone = list(model.children())[:-1]
model = torch.nn.Sequential(*backbone)

# train
# criterion = torch.nn.MSELoss()
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

log = []
for epoch in (range(EPOCHS)):
    loss_trn, log_trn = train(model, trn_loader, optimizer, criterion, device, log = True)
    loss_val = eval(model, val_loader, criterion, device)

    log.append([loss_trn, loss_val])
    plot_log(log_trn, loss_val = loss_val, epoch = epoch)
    torch.save(model, 'output/partial_{}.pt'.format(model_label))

print(log)
torch.save(model.cpu(), 'output/{}.pt'.format(model_label))
