
import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_log(model_label, log_trn, log_val, epoch):
    plt.figure();
    plt.plot(log_trn);
    plt.axhline(y = log_val, color = 'tab:orange');
    plt.title('trn: {}, val: {}'.format(np.mean(log_trn), log_val));
    plt.savefig('output/{}/loss_{}.png'.format(model_label, epoch));
    plt.close();

def export(model, model_label):
    print('exporting model')
    torch.save(model, 'output/{}/{}.pt'.format(model_label, model_label))
