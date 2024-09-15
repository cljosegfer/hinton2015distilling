
import os
import torch
import torch.nn as nn
import json

from tqdm import tqdm

from utils import plot_log

def export(model, model_label):
    print('exporting model')
    torch.save(model, 'output/{}/backbone/{}.pt'.format(model_label, model_label))

class Runner():
    def __init__(self, device, model, model_label):
        self.device = device
        self.model = model
        self.model_label = model_label
        if not os.path.exists('output/{}'.format(model_label)):
            os.makedirs('output/{}'.format(model_label))
        if not os.path.exists('output/{}/backbone'.format(model_label)):
            os.makedirs('output/{}/backbone'.format(model_label))
    
    def train(self, epochs, trn_loader, val_loader):
        self.model = self.model.to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)

        log = []
        minimo = 1e6
        best = self.model
        for epoch in range(epochs):
            print('-- epoch {}'.format(epoch))
            trn_log = self._train_loop(trn_loader, optimizer, criterion)['trn_log']
            val_log = self._eval_loop(val_loader, criterion)['val_log']
            # plot_log_epoch(self.model_label, trn_log, val_log, epoch)
            log.append([trn_log, val_log])
            if val_log < minimo:
                minimo = val_log
                best = self.model
                print('new checkpoint with val loss: {}'.format(minimo))
        plot_log(self.model_label + '/backbone', log)
        # export(self.model, self.model_label)
        self.model = best
        export(best, self.model_label)

    def _train_loop(self, loader, optimizer, criterion):
        # log = []
        log = 0
        self.model.train()
        for batch in tqdm(loader):
            x = batch['image'].to(self.device)
            h = batch['embedding'].to(self.device)

            hhat = self.model.forward(x)[:, :, 0, 0]
            loss = criterion(hhat, h)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log.append(loss.item())
            log += loss.item()
        # return {'trn_log': log}
        return {'trn_log': log / len(loader)}
    
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

