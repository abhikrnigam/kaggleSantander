import pandas as pd
import numpy as np
import torch


def getPredictions(loader, model, device):
    model.eval()
    saved_pred=[]
    true_labels=[]

    with torch.no_grad():
        for x,y in loader:
            x=x.to(device)
            y=y.to(device)
            score = model(x)
            saved_pred+= score.tolist()
            true_labels+= y.tolist()

    model.train()
    return saved_pred,true_labels

def get_submission(model,loader,test_ids,device):
    all_pred=[]
    model.eval()
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            score = model(x)
            prediction = score.float()
            all_pred+=prediction.tolist()

    model.train()
    df = pd.DataFrame({
        "ID_code" : test_ids.values,
        "target" : np.array(all_pred)
    })

    df.to_csv('output1.csv',index = False)