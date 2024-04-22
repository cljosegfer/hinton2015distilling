
import matplotlib.pyplot as plt
import numpy as np
import torch

from collections import OrderedDict

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

# from https://gist.github.com/the-bass/0bf8aaa302f9ba0d26798b11e4dd73e3
def load_backbone(model, backbone_path):
    backbone = torch.load(backbone_path)

    key_transformation = []
    for key in model.state_dict().keys():
        key_transformation.append(key)

    state_dict = backbone.state_dict()
    new_state_dict = OrderedDict()
    for i, (key, value) in enumerate(state_dict.items()):
        new_key = key_transformation[i]
        new_state_dict[new_key] = value

    log = model.load_state_dict(new_state_dict, strict = False)
    assert log.missing_keys == ['fc.weight', 'fc.bias']
    
    return {'model': model, 'log': log}
