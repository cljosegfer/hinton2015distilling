
# # setup
import torch

from dataloader import Cifar100
from runners.baseline import Runner
from utils import val_split
from hparams import EPOCHS, BATCH_SIZE, NUM_WORKERS

# # init
model_label = 'resnet101'
model = torch.hub.load('pytorch/vision:v0.10.0', model_label, pretrained = False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trn_ds = Cifar100(split = 'train')
sampler = val_split(N = trn_ds.__len__())
trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size = BATCH_SIZE, 
                                         sampler = sampler['trn_sampler'], num_workers = NUM_WORKERS)
val_loader = torch.utils.data.DataLoader(trn_ds, batch_size = BATCH_SIZE, 
                                         sampler = sampler['val_sampler'], num_workers = NUM_WORKERS)

tst_ds = Cifar100(split = 'val')
tst_loader = torch.utils.data.DataLoader(tst_ds, batch_size = BATCH_SIZE,
                                         shuffle = False, num_workers = NUM_WORKERS)

runner = Runner(device, model, model_label)

# # train
runner.train(EPOCHS, trn_loader, val_loader)

# # eval
acc = runner.acc(tst_loader)
