from tqdm import tqdm_notebook as tqdm
from IPython.display import clear_output
import matplotlib.pyplot as plt
import random

import torch
from torch.utils.data import Dataset
import torch.nn as nn

OBSERVED_DAYS = 64     # ~2 months
DAYS_TO_PREDICT_AFTER  = 10
DAYS_TO_PREDICT_BEFORE = 50
TESTING_DAYS = 1000

HEAVY_QUAKE_THRES = 3.5

class Dataset_RNN_Train(Dataset):

    def __init__(self, celled_data):
        self.data = celled_data[0:
                                (celled_data.shape[0] -
                                 TESTING_DAYS)]
        self.size = (self.data.shape[0] -
                     DAYS_TO_PREDICT_BEFORE)
        
        print ('self.data :', self.data.shape)
        print ('size      :', self.size)
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return (self.data[idx],
                torch.sum(self.data[(idx +
                                     DAYS_TO_PREDICT_AFTER):
                                    (idx +
                                     DAYS_TO_PREDICT_BEFORE)] > HEAVY_QUAKE_THRES,
                          dim=0,
                          keepdim=True).squeeze(0) > 0)
        
class Dataset_RNN_Train_different_xy(Dataset):

    def __init__(self, 
                 celled_data_x, 
                 celled_data_y,
                 testing_days, 
                 heavy_quake_thres,
                 days_to_predict_before, 
                 days_to_predict_after):
        
        self.heavy_quake_thres =  heavy_quake_thres
        self.days_to_predict_before = days_to_predict_before
        self.days_to_predict_after = days_to_predict_after
        
        self.data_x = celled_data_x[0:
                                (celled_data_x.shape[0] -
                                 testing_days)]
        self.data_y = celled_data_y[0:
                                (celled_data_y.shape[0] -
                                 testing_days)]
        
        self.size_x = (self.data_x.shape[0] -
                     self.days_to_predict_before)
        self.size_y = (self.data_y.shape[0] -
                     self.days_to_predict_before)
        
        print ('self.data_x :', self.data_x.shape, 'self.data_y :', self.data_y.shape)
        print ('size_x      :', self.size_x, 'size_y      :', self.size_y)
        
    def __len__(self):
        return self.size_x
    
    def __getitem__(self, idx):
        x = self.data_x[idx]
        y = torch.sum(self.data_y[(idx +
                                     self.days_to_predict_after):
                                    (idx +
                                     self.days_to_predict_before)] > self.heavy_quake_thres,
                          dim=0,
                          keepdim=True).squeeze(0) > 0
        
        return (x, y)
        
def train_RNN_full (RNN_cell,
                   device,
                   dataloader_train,
                   n_cycles=1,
                   learning_rate=0.0003,
                   earthquake_weight=1.,
                   lr_decay=1.,
                   start_lr_decay=0,
                   weight_decay=0):
    
    loss_massive = []
    
    RNN_cell.to(device)
    
    weights = torch.tensor([1., earthquake_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weights)
    
    i = 0
    for cycle in range(n_cycles):
        
        optimizer = torch.optim.Adam(RNN_cell.parameters(), lr=learning_rate, weight_decay=weight_decay)
        optimizer.zero_grad()
        
        hid_state = RNN_cell.init_state(batch_size=1, device=device)
        for data in dataloader_train:
            
            inputs = data[0].to(device)
            labels = data[1].to(device)
            
            hid_state, outputs = RNN_cell.forward(inputs, hid_state)
            
            loss = criterion(outputs, labels.squeeze(0).long())
            loss_massive.append(loss.item())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            if (type(hid_state) == tuple):
                for elem in hid_state:
                    elem.detach_()
            else:
                hid_state.detach_()
            
            if (i)%100==0:
                clear_output(True)
                print ("Done :", i, "/", dataloader_train.__len__() * n_cycles)
                plt.plot(loss_massive,label='loss')
                plt.legend()
                plt.show()
            i += 1
        if i >= start_lr_decay:
            learning_rate /= lr_decay

def train_RNN_part(RNN_cell,
                   device,
                   dataset_train,
                   n_cycles=1,
                   queue_lenght=1,
                   learning_rate=0.0003,
                   earthquake_weight=1.,
                   lr_decay=1.):
    
    loss_massive = []
    i = 0
    
    RNN_cell.to(device)
    
    weights = torch.tensor([1., earthquake_weight], dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weights)
    
    i = 0
    for cycle in range(n_cycles):
        
        optimizer = torch.optim.Adam(RNN_cell.parameters(), lr=learning_rate)
        optimizer.zero_grad()
        
        hid_state = RNN_cell.init_state(batch_size=1, device=device)
        start = random.randint(0, dataset_train.__len__() - queue_lenght)
        
        for t in range(start, start + queue_lenght):
            
            data = dataset_train[t]
            inputs = data[0].unsqueeze(0).to(device)
            labels = data[1].unsqueeze(0).to(device)
            
            hid_state, outputs = RNN_cell.forward(inputs, hid_state)
            
            loss = criterion(outputs, labels.squeeze(1).long())
            loss_massive.append(loss.item())
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            
            i += 1
            
        if (i)%queue_lenght==0:
            clear_output(True)
            print ("Done :", cycle, "/", n_cycles)
            plt.plot(loss_massive,label='loss')
            plt.legend()
            plt.show()
        
        learning_rate /= lr_decay