
# # setup
import torch
import os

from dataloader import Cifar100
from runners.pretrain import Runner
from utils import val_split
from hparams import EPOCHS, BATCH_SIZE, NUM_WORKERS

# # init
reflexo = 'resnet18'
origem = 'resnet34'
model_label = reflexo + origem

model = torch.hub.load('pytorch/vision:v0.10.0', reflexo, pretrained = False)
model = list(model.children())[:-1]
model = torch.nn.Sequential(*model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trn_ds = Cifar100(split = 'train', embedding_path = 'output/{}/H_trn.pt'.format(origem))
sampler = val_split(N = trn_ds.__len__())
trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size = BATCH_SIZE, 
                                         sampler = sampler['trn_sampler'], num_workers = NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(trn_ds, batch_size = BATCH_SIZE, 
                                         sampler = sampler['val_sampler'], num_workers = NUM_WORKERS)

# tst_ds = Cifar100(split = 'val', embedding_path = 'output/{}/H_tst.pt'.format(origem))
# tst_loader = torch.utils.data.DataLoader(tst_ds, batch_size = BATCH_SIZE,
#                                          shuffle = False, num_workers = NUM_WORKERS)

runner = Runner(device, model, model_label)

# # train
runner.train(EPOCHS, trn_loader, val_loader)