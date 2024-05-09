
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
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters())

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
        plot_log(self.model_label, log)
        # export(self.model, self.model_label)
        self.model = best
        export(best, self.model_label)

    def _train_loop(self, loader, optimizer, criterion):
        # log = []
        log = 0
        self.model.train()
        for batch in tqdm(loader):
            x = batch['image'].to(self.device)
            y = batch['label'].to(self.device)

            yhat = self.model.forward(x)
            loss = criterion(yhat, y)

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
                y = batch['label'].to(self.device)

                yhat = self.model.forward(x)
                loss = criterion(yhat, y)

                log += loss.item()
        return {'val_log': log / len(loader)}
    
    def acc(self, loader):
        num = 0
        den = 0
        self.model.to(self.device).eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                x = batch['image'].to(self.device)
                y = batch['label'].to(self.device)

                yhat = self.model.forward(x)
                _, predicted = torch.max(yhat.data, 1)
                den += y.size(0)
                num += (predicted == y).sum().item()
        log = {'acc': num / den}
        with open('output/{}/accuracy.json'.format(self.model_label), 'w') as file:
            json.dump(log, file)
        return log

    def synthesis(self, loader):
        backbone = list(self.model.children())[:-1]
        backbone = torch.nn.Sequential(*backbone)
        backbone = backbone.to(self.device).eval()

        H = torch.empty(size = [0])
        with torch.no_grad():
            for batch in tqdm(loader):
                x = batch['image'].to(self.device)

                h = backbone.forward(x)
                H = torch.cat((H, h[:, :, 0, 0].cpu()))
        return {'image features': H}
