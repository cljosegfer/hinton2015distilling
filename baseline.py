
import os
import torch
import numpy as np
import pandas as pd

from runners.baseline import BaselineRunner

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lim = 7

metrics = np.zeros(shape = (lim + 1, 2))
for depth in range(lim + 1):
    print('depth {}'.format(depth))
    runner = BaselineRunner(device = device, depth = depth)
    log = runner.run()
    # log = list(np.random.rand(10))
    mu = np.mean(log)
    sigma = np.std(log)
    print(mu, sigma)
    metrics[depth, 0] = mu
    metrics[depth, 1] = sigma
metrics = pd.DataFrame(metrics, columns = ['mean', 'std'])
metrics.to_csv('output/baseline/metrics.csv')