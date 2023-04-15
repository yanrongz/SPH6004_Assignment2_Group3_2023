# Author: Liang Jingyu
from torch.utils.data import Dataset, DataLoader
import torch
class ICUData(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        x = torch.tensor(self.data[index])
        y = torch.tensor(self.targets[index])
        return x, y