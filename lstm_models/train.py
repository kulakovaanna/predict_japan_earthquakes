from copy import deepcopy
import numpy as np
from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset
import torch.nn as nn
from sklearn.metrics import auc


from config import *

OBSERVED_DAYS = 1  # ~6 months
DAYS_TO_PREDICT_AFTER = 10
DAYS_TO_PREDICT_BEFORE = 50
TESTING_DAYS = 1000

HEAVY_QUAKE_THRES = 3.5


class Dataset_RNN_Train(Dataset):

    def __init__(self, celled_data):
        self.data = celled_data[0 : (celled_data.shape[0] - TESTING_DAYS)]
        self.size = self.data.shape[0] - DAYS_TO_PREDICT_BEFORE

        print("self.data :", self.data.shape)
        print("size      :", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        min_idx = max(0, idx-OBSERVED_DAYS)
        max_value = self.data[min_idx:idx].max()
        input_data = self.data[min_idx:idx].sum() / max_value
        return (
            input_data,
            torch.sum(
                self.data[
                    (idx + DAYS_TO_PREDICT_AFTER) : (idx + DAYS_TO_PREDICT_BEFORE)
                ]
                > HEAVY_QUAKE_THRES,
                dim=0,
                keepdim=True,
            ).squeeze(0)
            > 0,
        )


class Dataset_RNN_Train_different_xy(Dataset):

    def __init__(
        self,
        celled_data_x,
        celled_data_y,
        testing_days,
        heavy_quake_thres,
        days_to_predict_before,
        days_to_predict_after,
    ):

        self.heavy_quake_thres = heavy_quake_thres
        self.days_to_predict_before = days_to_predict_before
        self.days_to_predict_after = days_to_predict_after

        self.data_x = celled_data_x[0 : (celled_data_x.shape[0] - testing_days)]
        self.data_y = celled_data_y[0 : (celled_data_y.shape[0] - testing_days)]

        self.size_x = self.data_x.shape[0] - self.days_to_predict_before
        self.size_y = self.data_y.shape[0] - self.days_to_predict_before

        print("self.data_x :", self.data_x.shape, "self.data_y :", self.data_y.shape)
        print("size_x      :", self.size_x, "size_y      :", self.size_y)

    def __len__(self):
        return self.size_x

    def __getitem__(self, idx):
       
        start_idx = max(0, idx - OBSERVED_DAYS + 1)
        x = torch.sum(self.data_x[start_idx:idx + 1], dim=0)
        max_val = x.max()
        if max_val > 0:
            x = x / max_val
    
        y = (
            torch.sum(
                self.data_y[
                    (idx + self.days_to_predict_after) : (
                        idx + self.days_to_predict_before
                    )
                ]
                > self.heavy_quake_thres,
                dim=0,
                keepdim=True,
            ).squeeze(0)
            > 0
        )

        return (x, y)
    
class Dataset_RNN_Test(Dataset):
    def __init__(self, celled_data):
        self.data = celled_data[
            (celled_data.shape[0] - TESTING_DAYS) : (celled_data.shape[0])
        ]
        self.size = self.data.shape[0] - DAYS_TO_PREDICT_BEFORE

        print("self.data :", self.data.shape)
        print("size      :", self.size)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        min_idx = max(0, idx-OBSERVED_DAYS)
        max_value = self.data[min_idx:idx+1].max()
        input_data = self.data[min_idx:idx+1].sum() / max_value
        return (
            input_data,
            torch.sum(
                self.data[
                    (idx + DAYS_TO_PREDICT_AFTER) : (idx + DAYS_TO_PREDICT_BEFORE)
                ]
                > HEAVY_QUAKE_THRES,
                dim=0,
                keepdim=True,
            ).squeeze(0)
            > 0,
        )


class Dataset_RNN_Test_different_xy(Dataset):

    def __init__(
        self,
        celled_data_x,
        celled_data_y,
        testing_days,
        heavy_quake_thres,
        days_to_predict_before,
        days_to_predict_after,
    ):

        self.heavy_quake_thres = heavy_quake_thres
        self.days_to_predict_before = days_to_predict_before
        self.days_to_predict_after = days_to_predict_after

        self.data_x = celled_data_x[
            (celled_data_x.shape[0] - testing_days) : (celled_data_x.shape[0])
        ]
        self.data_y = celled_data_y[
            (celled_data_y.shape[0] - testing_days) : (celled_data_y.shape[0])
        ]

        self.size_x = self.data_x.shape[0] - self.days_to_predict_before
        self.size_y = self.data_y.shape[0] - self.days_to_predict_before

        print("self.data_x :", self.data_x.shape, "self.data_y :", self.data_y.shape)
        print("size_x      :", self.size_x, "size_y      :", self.size_y)

    def __len__(self):
        return self.size_x

    def __getitem__(self, idx):
        start_idx = max(0, idx - OBSERVED_DAYS + 1)
        x = torch.sum(self.data_x[start_idx:idx + 1], dim=0)
        max_val = x.max()
        if max_val > 0:
            x = x / max_val
        y = (
            torch.sum(
                self.data_y[
                    (idx + self.days_to_predict_after) : (
                        idx + self.days_to_predict_before
                    )
                ]
                > self.heavy_quake_thres,
                dim=0,
                keepdim=True,
            ).squeeze(0)
            > 0
        )

        return (x, y)

    
def get_target_pred(RNN_cell, device, dataloader_test):

    prediction = torch.zeros(
        dataloader_test.__len__(),
        N_CELLS_HOR,
        N_CELLS_VER,
        device=device,
        dtype=torch.float,
    )
    prediction.detach_()
    target = torch.zeros(
        dataloader_test.__len__(),
        N_CELLS_HOR,
        N_CELLS_VER,
        device=device,
        dtype=torch.float,
    )
    target.detach_()

    RNN_cell.to(device)

    hid_state = RNN_cell.init_state(batch_size=1, device=device)
    if type(hid_state) == tuple:
        for elem in hid_state:
            elem.detach_()
    else:
        hid_state.detach_()

    i = 0
    for data in tqdm(dataloader_test, leave=False):

        inputs = data[0].to(device)
        labels = data[1].to(device).float()

        hid_state, outputs = RNN_cell.forward(inputs, hid_state)

        prediction[i] = outputs[:, 1, :, :]
        target[i] = labels.squeeze(0)

        if type(hid_state) == tuple:
            for elem in hid_state:
                elem.detach_()
        else:
            hid_state.detach_()
        prediction.detach_()
        target.detach_()
        i += 1

    assert prediction.shape == target.shape
    prediction = prediction[10 : prediction.shape[0]]  # cutting peace of data because
    target = target[10 : target.shape[0]]  # hidden state might be not good

    return target, prediction


def my_precision_TPR_FPR(input, target, threshold, sample_weight=None, method="weight", device=torch.device("cpu")):
    """sample_weight is used only for fpr"""
    
    assert method in ["weight", "mask"], f"unknown method: {method}"
    
    if sample_weight is None:
        sample_weight = torch.ones_like(target).float()
        
    sample_weight = sample_weight.to(device)
    
    if method=="weight":
        TP = torch.sum(((input > threshold).float() * target).float())
        FP = torch.sum(((input > threshold).float() * (1 - target)).float() * sample_weight)
        FN = torch.sum(((~(input > threshold)).float() * target).float())
        TN = torch.sum(((~(input > threshold)).float() * (1 - target)).float() * sample_weight)
    elif method=="mask":
        sample_weight = (sample_weight > 0).float()
        
        TP = torch.sum(((input > threshold).float() * target).float() * sample_weight)
        FP = torch.sum(((input > threshold).float() * (1 - target)).float() * sample_weight)
        FN = torch.sum(((~(input > threshold)).float() * target).float() * sample_weight)
        TN = torch.sum(((~(input > threshold)).float() * (1 - target)).float() * sample_weight)

    return (TP, TP + FN), TP / (TP + FN), FP / (FP + TN)


def calc_roc(target, prediction, weights=None, method="weight", n_dots=500, device=torch.device("cpu")):
    fpr = []
    tpr = []
    trs = np.linspace(-0.0001, 1.02, n_dots)
    if weights is None:
        desc = "roc auc"
    else:
        desc = "weighted fpr roc auc"
    for t in tqdm(trs, desc=desc, leave=False):
        _, _tpr, _fpr = my_precision_TPR_FPR(prediction, target, t, sample_weight=weights, method=method, device=device)
        fpr.append(_fpr.cpu().numpy())
        tpr.append(_tpr.cpu().numpy())
       
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    mask = ((~np.isnan(fpr)) & (~np.isnan(tpr)))
    fpr = fpr[mask]
    tpr = tpr[mask]
    roc_auc = auc(fpr, tpr)
    
    return roc_auc, tpr, fpr

def train_RNN_full(
    RNN_cell,
    device,
    dataloader_train,
    dataloader_test,
    n_cycles=1,
    learning_rate=0.0003,
    earthquake_weight=1.0,
    non_earthquake_weight=1.0,
    lr_decay=1.0,
    start_lr_decay=0,
    weight_decay=0,
    mask=None,
    weight_mask=None,
    batch_size=1,
    min_best_epoch=-1,
):

    loss_massive = []
    test_metric_massive = []
    max_w_roc = -1
    best_cycle = -1
    best_model = None
    roc = None

    RNN_cell.to(device)

    weights = torch.tensor([non_earthquake_weight, earthquake_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weights, reduction='mean')

    i = 0
    fig, axes = [], []
    for cycle in range(n_cycles):

        optimizer = torch.optim.AdamW(
            RNN_cell.parameters(), lr=learning_rate, weight_decay=weight_decay, # amsgrad=True
        )
        optimizer.zero_grad()
        
        prediction = torch.zeros(
            dataloader_train.__len__(),
            N_CELLS_HOR,
            N_CELLS_VER,
            device=device,
            dtype=torch.float,
        )
        prediction.detach_()
        target = torch.zeros(
            dataloader_train.__len__(),
            N_CELLS_HOR,
            N_CELLS_VER,
            device=device,
            dtype=torch.float,
        )
        target.detach_()

        RNN_cell.train()
        hid_state = RNN_cell.init_state(batch_size=batch_size, device=device)
        cycle_loss = []
        for i_data, data in enumerate(dataloader_train):

            inputs = data[0].to(device)
            labels = data[1].to(device)

            hid_state, outputs = RNN_cell.forward(inputs, hid_state)
            
            loss = criterion(outputs, labels.squeeze(1).long())
            
            if i_data > 50:
                cycle_loss.append(loss.item())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if type(hid_state) == tuple:
                for elem in hid_state:
                    elem.detach_()
            else:
                hid_state.detach_()

            i += 1
        
           
        if i >= start_lr_decay:
            learning_rate *= lr_decay
            
        loss_massive.append(np.mean(cycle_loss))
        cycle_loss = []
        
        # test
        RNN_cell.eval()
        target, prediction = get_target_pred(
            RNN_cell=RNN_cell, device=device, dataloader_test=dataloader_test
        )
        roc_w, tpr_w, fpr_w = calc_roc(target, prediction, weights=weight_mask, method="weight", device=device)
        if roc_w > max_w_roc and cycle>min_best_epoch:
            roc, tpr, fpr = calc_roc(target, prediction, weights=mask, method="mask", device=device)
            max_w_roc = roc_w
            best_cycle = cycle
            best_model = deepcopy(RNN_cell)
            res_dict = {
                "roc": roc,
                "tpr": tpr.tolist(),
                "fpr": fpr.tolist(),
                "roc_w": roc_w,
                "tpr_w": tpr_w.tolist(),
                "fpr_w": fpr_w.tolist(),
            }
            
            
        test_metric_massive.append(roc_w)
        
        clear_output(True)
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
        axes[0].plot(loss_massive, label="train", color='orange')
        axes[0].set_title("loss")
        axes[0].set_xlabel("epoches")
        axes[0].set_ylabel("loss")
        axes[0].grid(True)
        
        axes[1].plot(test_metric_massive, label="test", color='blue')
        axes[1].set_title("roc auc weighted")
        axes[1].set_xlabel("epoches")
        axes[1].set_ylabel("roc auc")
        axes[1].grid(True)
        axes[1].legend()
        
        plt.tight_layout()
        plt.show()
        print(f"best epoch ({best_cycle}) with roc={roc}, w_roc={max_w_roc}")
        
    print(f"best epoch {best_cycle} with roc={roc}, w_roc={max_w_roc}")
        
    return best_model, res_dict
