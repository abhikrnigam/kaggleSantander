from math import ceil
import pandas as pd
import torch
from torch import float32
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split

def getData():
    # this is for the training dataset
    train_data = pd.read_csv("train.csv")
    y = train_data['target']
    X = train_data.drop(['ID_code', 'target'], axis=1)
    X_tensor = torch.tensor(X.values, dtype=float32)
    y_tensor = torch.tensor(y.values, dtype=float32)
    dataset = TensorDataset(X_tensor,y_tensor)
    train_ds, val_ds = random_split(dataset, [int(0.999*len(dataset)), ceil(0.001*len(dataset))])
    # this is for the test dataset
    test_data = pd.read_csv("test.csv")
    test_ids = test_data['ID_code']
    X = test_data.drop(['ID_code'], axis=1)
    X_tensor = torch.tensor(X.values, dtype=float32)
    y_tensor = torch.tensor(y.values, dtype=float32)
    test_ds = TensorDataset(X_tensor, y_tensor)
    return train_ds, val_ds, test_ds, test_ids

