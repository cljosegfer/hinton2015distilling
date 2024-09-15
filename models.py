
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, depth, xs = 1, ys = 10, hs = 25):
        super(CNN, self).__init__()
        self.depth = depth
        linear_in = hs * 40
        self.first = nn.Conv1d(in_channels = xs, out_channels = hs, kernel_size = 3, padding = 'same')
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hs, out_channels = hs, kernel_size = 3, padding = 'same') for u in range(self.depth)])

        self.linear = nn.Linear(linear_in, ys)

    def forward(self, x, verbose=False):
        x = x.view(-1, 1, x.shape[-1])
        h = self.first(x).relu()
        for u in range(self.depth):
            h = self.convs[u](h).relu()
        
        h = h.view(h.shape[0], -1)
        return self.linear(h)