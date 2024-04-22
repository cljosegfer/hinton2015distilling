
import os
import torch
import torch.nn as nn
import json

from tqdm import tqdm

from utils import plot_log, export

class Runner():
    def __init__(self, device, model, model_label):
        self.device = device
        self.model = model
        self.model_label = model_label
        if not os.path.exists('output/{}'.format(model_label)):
            os.makedirs('output/{}'.format(model_label))
    
    def train(self, epochs, trn_loader, val_loader):
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(epochs):
            print('-- epoch {}'.format(epoch))
            trn_log = self._train_loop(trn_loader, optimizer, criterion)['trn_log']
            val_log = self._eval_loop(val_loader, criterion)['val_log']
            plot_log(self.model_label, trn_log, val_log, epoch)
        export(self.model, self.model_label)

    def _train_loop(self, loader, optimizer, criterion):
        log = []
        self.model.train()
        for batch in tqdm(loader):
            x = batch['image'].to(self.device)
            h = batch['embedding'].to(self.device)

            hhat = self.model.forward(x)[:, :, 0, 0]
            loss = criterion(hhat, h)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            log.append(loss.item())
        return {'trn_log': log}
    
    def _eval_loop(self, loader, criterion):
        log = 0
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                x = batch['image'].to(self.device)
                h = batch['embedding'].to(self.device)

                hhat = self.model.forward(x)[:, :, 0, 0]
                loss = criterion(hhat, h)

                log += loss.item()
        return {'val_log': log / len(loader)}

