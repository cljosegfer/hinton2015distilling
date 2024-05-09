
# # setup
import torch
import os

from dataloader import Cifar100
from runners.pretrain import Runner

# # init
reflexo = 'resnet18'
origem = 'resnet34'
model_label = reflexo + origem

model = torch.hub.load('pytorch/vision:v0.10.0', reflexo, pretrained = False)
model = list(model.children())[:-1]
model = torch.nn.Sequential(*model)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 100
BATCH_SIZE = 128
NUM_WORKERS = 6

trn_ds = Cifar100(split = 'train', embedding_path = 'output/{}/H_trn.pt'.format(origem))
trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size = BATCH_SIZE,
                                          shuffle = True, num_workers = NUM_WORKERS)

val_ds = Cifar100(split = 'val', embedding_path = 'output/{}/H_val.pt'.format(origem))
val_loader = torch.utils.data.DataLoader(val_ds, batch_size = BATCH_SIZE,
                                         shuffle = False, num_workers = NUM_WORKERS)

runner = Runner(device, model, model_label)

# # train
runner.train(EPOCHS, trn_loader, val_loader)

if not os.path.exists('output/{}/backbone'.format(model_label)):
    os.makedirs('output/{}/backbone'.format(model_label))

where = 'output/{}'.format(model_label)
togo = 'output/{}/backbone'.format(model_label)

for file in os.listdir(where):
    if file == 'backbone':
        continue
    os.rename(os.path.join(where, file), os.path.join(togo, file))
