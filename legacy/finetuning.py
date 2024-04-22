
# setup

import torch
import os
import matplotlib.pyplot as plt

from utils import Imagenet_xy, train, eval, model_select, plot_log

from tqdm import tqdm
from collections import OrderedDict

BATCH_SIZE = 128
EPOCHS = 10

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists('output'):
    os.makedirs('output')

# data
trn = Imagenet_xy('train')
val = Imagenet_xy('val')

trn_loader = torch.utils.data.DataLoader(trn, batch_size = BATCH_SIZE, shuffle = True)
val_loader = torch.utils.data.DataLoader(val, batch_size = BATCH_SIZE)

# model
origem = 'rn18'
espelhado = 'rn34'
model_label = origem + espelhado
backbone = torch.load('output/l1_{}.pt'.format(model_label))

# model = model_select(origem, pretrained = False)

# key_transformation = []
# for key in model.state_dict().keys():
#     key_transformation.append(key)

# # from https://gist.github.com/the-bass/0bf8aaa302f9ba0d26798b11e4dd73e3

# state_dict = backbone.state_dict()
# new_state_dict = OrderedDict()

# for i, (key, value) in enumerate(state_dict.items()):
#     new_key = key_transformation[i]
#     new_state_dict[new_key] = value

# log = model.load_state_dict(new_state_dict, strict = False)
# assert log.missing_keys == ['fc.weight', 'fc.bias']
# model = torch.load('output/partial_{}_finetuned.pt'.format(model_label))
model = model_select(origem, pretrained = False)

model = model.to(device)

# train
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

log = []
for epoch in (range(EPOCHS)):
    loss_trn, log_trn = train(model, trn_loader, optimizer, criterion, device, log = True)
    loss_val = eval(model, val_loader, criterion, device)

    log.append([loss_trn, loss_val])
    plot_log(log_trn, loss_val = loss_val, epoch = epoch)
    torch.save(model, 'output/partial_{}_normal.pt'.format(origem))

print(log)
torch.save(model.cpu(), 'output/{}_normal.pt'.format(origem))
