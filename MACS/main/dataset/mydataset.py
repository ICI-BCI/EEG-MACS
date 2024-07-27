from torch.utils.data import Dataset
import torch
import numpy as np
class MyTrainDataset(Dataset):
    def __init__(self, data,labels):
        self.data = data.float()
        labels = list(map(int, labels))
        self.targets = labels
       

    def __getitem__(self, idx):
        x=self.data[idx]
        y=self.targets[idx]
        
        return x,y,idx


    def __len__(self):
        return len(self.data)
    
class MyTestDataset(Dataset):
    def __init__(self, data,labels,transform_test):
        self.data = data
        labels = list(map(int, labels))
        self.targets = labels
        self.transform=transform_test

    def __getitem__(self, idx):
  
        x=self.data[idx]
        y=self.targets[idx]

        return x,y


    def __len__(self):
        return len(self.data)