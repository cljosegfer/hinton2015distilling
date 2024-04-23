
# # setup
import torch
import os

from dataloader import Cifar10
from runners.baseline import Runner

# # init
model_label = 'resnet101'
model = torch.hub.load('pytorch/vision:v0.10.0', model_label, pretrained = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 6

trn_ds = Cifar10(split = 'train')
trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size = BATCH_SIZE,
                                          shuffle = True, num_workers = NUM_WORKERS)

val_ds = Cifar10(split = 'val')
val_loader = torch.utils.data.DataLoader(val_ds, batch_size = BATCH_SIZE,
                                         shuffle = False, num_workers = NUM_WORKERS)

runner = Runner(device, model, model_label)

# # train
runner.train(EPOCHS, trn_loader, val_loader)

# # eval
acc = runner.acc(val_loader)
