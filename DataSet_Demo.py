import torch
from torch.utils.data import Dataset
class subDataset(Dataset):
    def __init__(self, Data, Label):
        self.Data = Data
        self.Label = Label
    def __len__(self):
        return len(self.Data)
    def __getitem__(self, index):
        data = self.Data[index]
        label = self.Label[index]
        return data, label