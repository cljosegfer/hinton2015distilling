
import torch

from mnist1d.data import make_dataset, get_dataset_args
from sklearn.model_selection import train_test_split, KFold
from torch.utils.data import Dataset

class MNIST1Ddb():
    def __init__(self):
        default_args = get_dataset_args()
        self.data = make_dataset(default_args)

    def split(self):
        x, y = self.data['x'], self.data['y']
        x_trn, x_val, y_trn, y_val = train_test_split(x, y, test_size = 0.1)
        x_tst, y_tst = self.data['x_test'], self.data['y_test']
        return {'x': x_trn, 'y': y_trn}, {'x': x_val, 'y': y_val}, {'x': x_tst, 'y': y_tst}
    
    def kfold(self, k = 10):
        kf = KFold(n_splits = k, shuffle = True, random_state = 0)
        x, y = self.data['x'], self.data['y']
        return kf.split(x, y)

class MNIST1D(Dataset):
    def __init__(self, db, device = 'cpu'):
        self.db = db
        self.db['x'] = torch.Tensor(self.db['x']).to(device)
        self.db['y'] = torch.LongTensor(self.db['y']).to(device)

    def __len__(self):
        return self.db['x'].shape[0]
    
    def __getitem__(self, idx):
        return {'x': self.db['x'][idx, :], 'y': self.db['y'][idx]}