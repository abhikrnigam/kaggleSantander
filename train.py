from math import ceil
import pandas as pd
import torch
import torch.nn as nn
from sklearn import metrics
import dataset
import torch.optim as optim
from utils import getPredictions
from torch import float32
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class NN(nn.Module):
    def __init__(self, input_size):
        super(NN, self).__init__()

        self.net = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Linear(input_size, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x)).view(-1)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NN(input_size=200).to(DEVICE)
## Karpathy constant lr=3e-4
optimiser = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
loss_func = nn.BCELoss()
train_ds, val_ds, test_ds, test_ids = dataset.getData()
train_loader = DataLoader(train_ds,batch_size=1024, shuffle=True)
val_loader = DataLoader(val_ds, batch_size = 1024)
test_loader = DataLoader(test_ds,batch_size=1024)

for epoch in range(15):
    probabilities , true = getPredictions(val_loader,model,device = DEVICE)
    print(f"validation roc: { metrics.roc_auc_score(true,probabilities)}")
    # use the below statment for checking for only one batch if it goes correct remove it
    #data, targets = next(iter(train_loader))
    for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)

            # forward
            scores = model(data)
            loss = loss_func(scores, targets)
            #print(loss)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

from utils import get_submission
get_submission(model,test_loader,test_ids,DEVICE)