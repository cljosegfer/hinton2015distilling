
# # setup
import torch
import os

from dataloader import Cifar100
from runners.baseline import Runner
from utils import load_backbone

# # init
reflexo = 'resnet18'
origem = 'resnet34'
model_label = reflexo + origem

model = torch.hub.load('pytorch/vision:v0.10.0', reflexo, pretrained = False)
model = load_backbone(model, 'output/{}/backbone/{}.pt'.format(model_label, model_label))['model']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 128
NUM_WORKERS = 6

trn_ds = Cifar100(split = 'train')
trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size = BATCH_SIZE,
                                          shuffle = True, num_workers = NUM_WORKERS)

val_ds = Cifar100(split = 'val')
val_loader = torch.utils.data.DataLoader(val_ds, batch_size = BATCH_SIZE,
                                         shuffle = False, num_workers = NUM_WORKERS)

runner = Runner(device, model, model_label)

# # train
runner.train(EPOCHS, trn_loader, val_loader)

# # eval
acc = runner.acc(val_loader)
