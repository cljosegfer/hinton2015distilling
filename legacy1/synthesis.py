
# # setup
import torch

from dataloader import Cifar100
from runners.toy import Runner
from hparams import EPOCHS, BATCH_SIZE, NUM_WORKERS

# # init
model_label = 'resnet34'
model = torch.load('output/{}/{}.pt'.format(model_label, model_label))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

trn_ds = Cifar100(split = 'train')
trn_loader = torch.utils.data.DataLoader(trn_ds, batch_size = BATCH_SIZE,
                                          shuffle = False, num_workers = NUM_WORKERS)

tst_ds = Cifar100(split = 'val')
tst_loader = torch.utils.data.DataLoader(tst_ds, batch_size = BATCH_SIZE,
                                         shuffle = False, num_workers = NUM_WORKERS)

runner = Runner(device, model, model_label)

# # synthesis
H_trn = runner.synthesis(trn_loader)['image features']
torch.save(H_trn, 'output/{}/H_trn.pt'.format(model_label))

H_tst = runner.synthesis(tst_loader)['image features']
torch.save(H_tst, 'output/{}/H_tst.pt'.format(model_label))