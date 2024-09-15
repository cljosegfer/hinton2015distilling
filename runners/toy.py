
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from models import CNN
from dataset import MNIST1Ddb, MNIST1D

class BaselineRunner():
    def __init__(self, device, depth):
        self.device = device
        self.depth = depth
        self.model = CNN(depth = self.depth)
        self.db = MNIST1Ddb()
        
        db_trn, db_val, db_tst = self.db.split()
        self.ds_trn = MNIST1D(db = db_trn, device = device)
        self.ds_val = MNIST1D(db = db_val, device = device)
        self.ds_tst = MNIST1D(db = db_tst, device = device)
    
    def run(self, epochs = 200, lr = 1e-2, bs = 100, plot = True):
        self.model = self.model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr = lr)

        trn_loader = DataLoader(self.ds_trn, batch_size = bs, shuffle = True)
        val_loader = DataLoader(self.ds_val, batch_size = bs, shuffle = False)
        tst_loader = DataLoader(self.ds_tst, batch_size = bs, shuffle = False)

        self.loss_trn = []
        self.loss_val = []
        self.loss_tst = []
        self.acc_val = []
        self.acc_tst = []

        for epoch in tqdm(range(epochs)):
            self.loss_trn.append(self.train(trn_loader, optimizer, criterion))
            
            log, acc = self.eval(val_loader, criterion)
            self.loss_val.append(log)
            self.acc_val.append(acc)
            
            log, acc = self.eval(tst_loader, criterion)
            self.loss_tst.append(log)
            self.acc_tst.append(acc)
        
        if plot:
            self.plot()
    
    def train(self, loader, optimizer, criterion):
        self.model.train()
        
        log = 0
        for batch in (loader):
            x, y = batch['x'], batch['y']
            
            yhat = self.model.forward(x)
            loss = criterion(yhat, y)
            loss.backward(); optimizer.step(); optimizer.zero_grad()
            
            log += loss.item()
        log /= loader.dataset.__len__()
        return log
    
    def eval(self, loader, criterion):
        self.model.eval()

        log = 0
        acc = 0
        with torch.no_grad():
            for batch in (loader):
                x, y = batch['x'], batch['y']
            
                yhat = self.model.forward(x)
                loss = criterion(yhat, y)
                
                log += loss.item()
                acc += np.sum(yhat.argmax(-1).cpu().numpy() == y.cpu().numpy())
        log /= loader.dataset.__len__()
        acc /= loader.dataset.__len__()
        return log, acc
    
    def plot(self):
        fig, axes = plt.subplots(1, 2, figsize = (14, 4))

        axes[0].plot(self.loss_val, label = 'val');
        axes[0].plot(self.loss_tst, label = 'tst');
        axes[0].plot(self.loss_trn, label = 'trn');
        axes[0].legend();
        
        axes[1].plot(self.acc_val, label = 'val');
        axes[1].plot(self.acc_tst, label = 'tst');
        axes[1].legend();

        # minimo = np.min(self.loss_val)
        # best = np.argmin(self.loss_val)
        # axes[0].axhline(y = minimo, color = 'black', linestyle = 'dashed');
        # print(minimo, best, self.loss_tst[best])
        print(self.loss_val[-1], self.loss_tst[-1], self.loss_trn[-1])
        
        # maximo = np.max(self.acc_val)
        # best = np.argmax(self.acc_tst)
        # axes[1].axhline(y = maximo, color = 'black', linestyle = 'dashed');
        # print(maximo, best, self.acc_tst[best], self.acc_tst[-1])
        print(self.acc_val[-1], self.acc_tst[-1])