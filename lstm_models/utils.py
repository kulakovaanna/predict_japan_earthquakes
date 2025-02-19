import os
import math as m
import numpy as np
from datetime import datetime
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import auc

from train import *
from model import *
from config import *


def get_rates(prediction, target, threshold):
    fnr = (
        torch.sum(((prediction < threshold).float() * target).float()) / 
        torch.sum(target).float()
    )
    alram_rate = torch.sum(((prediction > threshold).float()).float())
    alram_rate = torch.sum( (prediction > threshold).float() ) / torch.sum(torch.ones_like(prediction))

    return fnr, alram_rate

def preprocess_time(year, month, day, hour, minute, s):
    """
    get time in format "1990-01-01T00:00:00Z"

    Args:
        year (_type_): _description_
        month (_type_): _description_
        day (_type_): _description_
        hour (_type_): _description_
        minute (_type_): _description_
        s (_type_): _description_

    Returns:
        str: formatted time
    """
    f0 = lambda x: f"0{int(x)}" if x < 10 else str(int(x))
    
    year = str(int(year))
    month = f0(month)
    day = f0(day)
    hour = f0(hour)
    minute = f0(minute)
    s = f0(s)
    
    return f"{year}-{month}-{day}T{hour}:{minute}:{s}Z"

def get_formatted_dataset(df, min_magnitude=0.):
    """
    formatting dataset for earthquake_jp preprocess() function.

    Args:
        df (pd.DataFrame): earthquakes dataset
        
        dataset in format:
        YYYY  MM  DD  HH  mm  ssss     LAT     LONG  Depth  Class
        0  2004   3   6  18   4   9.0  36.360  136.571    6.2    0.7
        1  2004   3   6  18   5  40.0  37.007  138.709    9.8    0.7
        
    Returns:
        pd.DataFrame: formatted dataset with columns ["time", "longitude", "latitude", "magnitude"]
    """
    
    df = df[df["Class"] > min_magnitude].reset_index(drop=True)
    df["time"] = df[["YYYY", "MM", "DD", "HH", "mm", "ssss"]].apply(
        lambda x: preprocess_time(*x),
        axis = 1
    )
    df = df.rename(
        columns={
            "LAT": "latitude", 
            "LONG": "longitude", 
            "Class": "magnitude"}
        )
    df["place"] = "Japan"
    df = df[["time", "longitude", "latitude", "magnitude", "place"]]
    
    return df

def create_celled_data(lon, lat, n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, bbox=BBOX, density=False, transpose=False):
    """the function creates celled data (maps).

    Args:
        lon (_type_): list of longtitude
        lat (_type_): list of latitude
        n_cells_hor (_type_): number of cells for horizontal ax
        n_cells_ver (_type_): number of cells for vertical ax
        bbox (_type_): bounding box for map in lon, lat coordinates
        density (bool, optional): if True returns density maps [0., ..., 1.]. 
            else returns a binary map. Defaults to True.

    Returns:
        celled_data, x_arr, y_arr: map, arrays of longtitude and latitude for the map.
    """
    LEFT_BORDER = bbox[0]
    RIGHT_BORDER = bbox[1]
    DOWN_BORDER = bbox[2]
    UP_BORDER = bbox[3]

    celled_data = np.zeros([n_cells_hor, n_cells_ver])

    cell_size_hor = (RIGHT_BORDER - LEFT_BORDER) / n_cells_hor
    cell_size_ver = (UP_BORDER - DOWN_BORDER) / n_cells_ver

    x = lon
    y = lat

    mask = (
        (x > LEFT_BORDER) &
        (x < RIGHT_BORDER) &
        (y > DOWN_BORDER) &
        (y < UP_BORDER)
    )
    x = x[mask]
    y = y[mask]

    x = ((x-LEFT_BORDER) / cell_size_hor).astype(int)
    y = ((y-DOWN_BORDER) / cell_size_ver).astype(int)

    if density:
        for x_i, y_i in zip(x, y):
            celled_data[x_i, y_i] += 1.
        celled_data = celled_data / np.max(celled_data)
    else:
        celled_data[x, y] = 1
        
    if transpose:
        celled_data = celled_data.transpose()[::-1]
    
    x_arr = np.zeros(celled_data.shape[0])
    y_arr = np.zeros(celled_data.shape[1])
    for i in range(celled_data.shape[0]):
        for j in range(celled_data.shape[1]):
            x_arr[i] = i*cell_size_hor + LEFT_BORDER
            y_arr[j] = j*cell_size_ver + DOWN_BORDER

    return celled_data, x_arr, y_arr


def plot_roc(target, prediction, weights=None, method="weight", src_label="", n_dots=501, device=torch.device("cpu")):   
    assert prediction.shape == target.shape
    assert method in ["weight", "mask"], f"unknown method: {method}"

    label = "roc auc"
    if src_label != "":
        label = " | ".join([label, src_label])
    
    fpr = []
    tpr = []
    n = []
    N = []
    trs = np.linspace(-0.0001, 1.02, n_dots)
    for t in tqdm(trs, desc=label, leave=False):
        n_and_N, _tpr, _fpr = my_precision_TPR_FPR(prediction, target, t, method=method, sample_weight=weights, device=device)
        fpr.append(_fpr.cpu().numpy())
        tpr.append(_tpr.cpu().numpy())
        n.append(float(n_and_N[0].cpu().numpy()))
        N.append(float(n_and_N[1].cpu().numpy()))
       
    fpr = np.asarray(fpr)
    tpr = np.asarray(tpr)
    mask = ((~np.isnan(fpr)) & (~np.isnan(tpr)))
    fpr = fpr[mask]
    tpr = tpr[mask]
    roc_auc = auc(fpr, tpr)
    roc_auc = round(roc_auc, 4)
    print(roc_auc)

    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.plot(range(2), range(2), 'grey', ls='--')
    ax.plot(fpr, tpr, c="b")
    plt.ylabel("tpr")
    plt.xlabel("fpr")
    plt.grid(alpha=0.4)
    plt.title(label)
    plt.show()
    
    return roc_auc, fpr, tpr, n, N

def plot_error_diagram(target, prediction, weights=None, src_label="", n_dots=501):   
    assert prediction.shape == target.shape

    label = "error diagram"
    if src_label != "":
        label = " | ".join([label, src_label])
    
    trs = np.linspace(-0.0001, 1.02, n_dots)
    fnr = []
    alarm = []
    for t in tqdm(trs, desc=label, leave=False):
        _fnr, _alarm = get_rates(prediction, target, t)
        fnr.append(_fnr.cpu().numpy())
        alarm.append(_alarm.cpu().numpy())
       
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.plot(range(2), range(2)[::-1], 'grey', ls='--')
    ax.plot(alarm, fnr, c="b")
    plt.ylabel("earthquake missing rate (fnr)")
    plt.xlabel("alarm rate")
    plt.grid(alpha=0.4)
    plt.title(label)
    plt.show()
    
def check_quality_diagram(target, prediction, n_dots=501, masked=False, weighted=False, weights=None):
    
    if masked:
        mask = weights
        prediction = prediction[:, mask > 0.0]
        target = target[:, mask > 0.0]
        if weighted and weights is not None:
            weights = weights[:, mask > 0.0]
    l = ""
    if masked:
        l += "masked"
        
    plot_error_diagram(target, prediction, weights=None, src_label=l)

    
def check_quality(target, prediction, n_dots=501, method="weight", weights=None, device=torch.device("cpu")):
    assert method in ["weight", "mask"], f"unknown method: {method}"
    
    l = ""
    if method=="weight" and weights is not None:
        l += "weighted fpr"
    elif method=="mask" and weights is not None:
        l += "masked"

    roc_auc, fpr, tpr, n, N = plot_roc(target, prediction, method=method, weights=weights, src_label=l, device=device)
    
    return roc_auc, fpr, tpr, n, N


def read_celled_data(pathname, n_cells_hor, n_cells_ver):
    return torch.load(pathname + str(n_cells_hor) + "x" + str(n_cells_ver))


class pipeline:
    def __init__(
        self,
        celled_data_x_path,
        celled_data_y_path,
        celled_data_path_for_freq_map,
        n_cells_hor,
        n_cells_ver,
        model_name,
        testing_days,
        heavy_quake_thres,
        days_to_predict_before,
        days_to_predict_after,
        embedding_size,
        hidden_state_size,
        n_cycles,
        learning_rate,
        lr_decay,
        start_lr_decay,
        weight_decay,
        earthquake_weight,
        device,
        model_type = "lstm",
        batch_size=1,
        non_earthquake_weight=1.0,
        min_best_epoch=-1,
        shuffle=False,
        num_workers=10,
        loss_mask=False,
        retrain=False
    ):

        self.retrain = retrain
        self.celled_data_x_path = celled_data_x_path
        self.celled_data_y_path = celled_data_y_path
        self.celled_data_path_for_freq_map = celled_data_path_for_freq_map
        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver
        self.model_type = model_type
        self.min_best_epoch = min_best_epoch

        if model_name is None:
            current_date_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
            self.model_name = f"{model_type}_{current_date_time}"
        else:
            self.model_name = model_name
            
        self.testing_days = testing_days
        self.heavy_quake_thres = heavy_quake_thres
        self.mask = self.get_weights_map(trs=self.heavy_quake_thres)[0, ...]
        self.mask = (self.mask > 0).float()
        self.weight_mask = self.get_weights_map(trs=3.0)[0, ...]
        
        self.days_to_predict_before = days_to_predict_before
        self.days_to_predict_after = days_to_predict_after
        self.embedding_size = embedding_size
        self.hidden_state_size = hidden_state_size
        self.n_cycles = n_cycles
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.start_lr_decay = start_lr_decay
        self.weight_decay = weight_decay
        self.earthquake_weight = earthquake_weight
        self.non_earthquake_weight = non_earthquake_weight
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.loss_mask = loss_mask
        self.density_map = self.get_weights_map()[0, ...]
        self.res_dict = None
        

    def get_celled_data_x_y(self):
        self.celled_data_x = read_celled_data(
            self.celled_data_x_path, self.n_cells_hor, self.n_cells_ver
        )
        self.celled_data_y = read_celled_data(
            self.celled_data_y_path, self.n_cells_hor, self.n_cells_ver
        )

        if self.celled_data_x.shape[1] == 1:
            self.two_maps_flag = False
        elif self.celled_data_x.shape[1] == 2:
            self.two_maps_flag = True

    def get_freq_map(self):
        freq_map_celled_data = read_celled_data(
            self.celled_data_path_for_freq_map, self.n_cells_hor, self.n_cells_ver
        )
        self.freq_map = (
            (freq_map_celled_data > self.heavy_quake_thres).float().mean(dim=0)
        )
        
        return self.freq_map

        
    def get_weights_map(self, trs=3.0, norm=True):
        freq_map_celled_data = read_celled_data(
            self.celled_data_path_for_freq_map, self.n_cells_hor, self.n_cells_ver
        )
        weights_map = (
            (freq_map_celled_data > trs).float().sum(dim=0)
        )
        if norm:
            return weights_map/torch.max(weights_map)
        else: 
            return weights_map

    def get_train_test_datasets(self):
        self.dataset_train = Dataset_RNN_Train_different_xy(
            celled_data_x=self.celled_data_x,
            celled_data_y=self.celled_data_y,
            testing_days=self.testing_days,
            heavy_quake_thres=self.heavy_quake_thres,
            days_to_predict_before=self.days_to_predict_before,
            days_to_predict_after=self.days_to_predict_after,
        )

        self.dataset_test = Dataset_RNN_Test_different_xy(
            celled_data_x=self.celled_data_x,
            celled_data_y=self.celled_data_y,
            testing_days=self.testing_days,
            heavy_quake_thres=self.heavy_quake_thres,
            days_to_predict_before=self.days_to_predict_before,
            days_to_predict_after=self.days_to_predict_after,
        )

    def get_train_test_dataloaders(self):
        self.dataloader_train = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
        self.dataloader_test = DataLoader(
            self.dataset_test,
            batch_size=1,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def save_model(self):
        if not os.path.exists("../data/model"):
            os.mkdir("../data/model")
        torch.save(self.model.state_dict(), f"../data/model/{self.model_name}")

    def init_model(self):
        if self.model_type=="gru" and self.two_maps_flag:
            self.model = GRUCell_density(
                frequency_map=self.freq_map,
                embedding_size=self.embedding_size,
                hidden_state_size=self.hidden_state_size,
                n_cells_hor=self.n_cells_hor,
                n_cells_ver=self.n_cells_ver,
                device=self.device,
            )
        elif self.model_type=="gru" and self.two_maps_flag == False:
            self.model = GRUCell(
                frequency_map=self.freq_map,
                embedding_size=self.embedding_size,
                hidden_state_size=self.hidden_state_size,
                n_cells_hor=self.n_cells_hor,
                n_cells_ver=self.n_cells_ver,
                device=self.device,
            )
        elif self.model_type=="lstm" and self.two_maps_flag == False:
            self.model = LSTMCell(
                frequency_map=self.freq_map,
                embedding_size=self.embedding_size,
                hidden_state_size=self.hidden_state_size,
                n_cells_hor=self.n_cells_hor,
                n_cells_ver=self.n_cells_ver,
                device=self.device,
            )

        elif self.model_type=="lstm" and self.two_maps_flag == True:
            self.model = LSTMCell_density(
                frequency_map=self.freq_map,
                embedding_size=self.embedding_size,
                hidden_state_size=self.hidden_state_size,
                n_cells_hor=self.n_cells_hor,
                n_cells_ver=self.n_cells_ver,
                device=self.device,
            )

    def import_model(self):
        weights_path = f"../data/model/{self.model_name}"
        if not os.path.exists(weights_path):
            RaiseException(f"'{weights_path}' does not exist")
        else:
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        
    def train(self, weight_mask=None, mask=None):
        self.model.train()
        self.model, _ = train_RNN_full(
            self.model,
            self.device,
            self.dataloader_train,
            self.dataloader_test,
            n_cycles=self.n_cycles,
            learning_rate=self.learning_rate,
            earthquake_weight=self.earthquake_weight,
            non_earthquake_weight = self.non_earthquake_weight,
            lr_decay=self.lr_decay,
            start_lr_decay=self.start_lr_decay,
            weight_decay=self.weight_decay,
            weight_mask=weight_mask,
            mask=mask,
            batch_size=self.batch_size,
            min_best_epoch=self.min_best_epoch
        )

        self.save_model()

    def validate(self, target, prediction, method="weight", weights=None, val_type='roc'):
        assert method in ["weight", "mask"], f"unknown method: {method}"
        
        if weights is None:
            RaiseException("no weights or mask provided")
        
        self.model.eval()
        
        if val_type=='roc':
            ROC_AUC, fpr, tpr, n, N = check_quality(
                target, prediction, 
                method=method,
                weights=weights,
                device=self.device
            )
            return ROC_AUC, fpr, tpr, n, N
#         elif val_type=='diagram':
#             check_quality_diagram(
#                 target, prediction,
#                 masked=masked, 
#                 weighted=weighted, 
#                 weights=weights,
#             )
        

    def __call__(self):
        print("initializing datasets, dataloaders, model ... ")
        self.get_celled_data_x_y()
        self.get_freq_map()
        density_map = self.get_weights_map()[0, ...]
        
        print(f"two_maps_flag: {self.two_maps_flag}")
        self.get_train_test_datasets()
        self.get_train_test_dataloaders()
        self.init_model()
        
        weights_path = f"../data/model/{self.model_name}"
        print(weights_path, os.path.exists(weights_path))
        if os.path.exists(weights_path) and self.retrain==False:
            self.import_model()
            print("model imported")
        else:
            print("\ntraining ...")
            self.train(weight_mask=self.weight_mask, mask=self.mask)

        print("\nvalidating ...")
        
        target, prediction = get_target_pred(
            RNN_cell=self.model, device=self.device, dataloader_test=self.dataloader_test
        )

        roc, tpr, fpr, n, N = self.validate(target, prediction, method="mask", weights=self.mask)
        roc_w, tpr_w, fpr_w, n_w, N_w = self.validate(target, prediction, method="weight", weights=self.weight_mask)
        
        return {
            "model": self.model_name,
            "roc": roc,
            "tpr": tpr,
            "fpr": fpr,
            "n": n,
            "N": N,
            "roc_w": roc_w,
            "tpr_w": tpr_w,
            "fpr_w": fpr_w,
            "n_w": n_w,
            "N_w": N_w,
        }


def plot_density(density_map, map_path=MAP_PATH, figsize=FIGSIZE, bbox=BBOX):
    map_img = plt.imread(map_path)
    fig, ax = plt.subplots(
        figsize=(figsize, figsize * map_img.shape[0] / map_img.shape[1]), dpi=DPI
    )
    ax.imshow(map_img, zorder=0, extent=bbox)
    plt.imshow(
        density_map, alpha=0.7, interpolation="bicubic", extent=bbox, cmap="seismic"
    )


def get_coords_from_map(
    celled_data, n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, bbox=BBOX
):
    LEFT_BORDER = 0
    RIGHT_BORDER = 2000
    DOWN_BORDER = 0
    UP_BORDER = 2500

    cell_size_hor = (RIGHT_BORDER - LEFT_BORDER) / n_cells_hor
    cell_size_ver = (UP_BORDER - DOWN_BORDER) / n_cells_ver

    x_arr = np.zeros(celled_data.shape[0])
    y_arr = np.zeros(celled_data.shape[1])
    for i in range(celled_data.shape[0]):
        for j in range(celled_data.shape[1]):
            x_arr[i] = i * cell_size_hor + LEFT_BORDER
            y_arr[j] = j * cell_size_ver + DOWN_BORDER

    return x_arr, y_arr


def cartesian_to_spherical(x, y):
    lat = (y / (m.pi / 180.0 * EARTH_RADIUS)) + ORIGIN_LATITUDE
    lon = (
        x / (m.pi / 180.0 * EARTH_RADIUS * m.cos(lat * m.pi / 180.0))
    ) + ORIGIN_LONGITUDE

    return lon, lat


def get_coords(
    celled_data, n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, bbox=BBOX
):
    LEFT_BORDER = 0
    RIGHT_BORDER = 2000
    DOWN_BORDER = 0
    UP_BORDER = 2500

    LEFT_BORDER_TRG = bbox[0]
    RIGHT_BORDER_TRG = bbox[1]
    DOWN_BORDER_TRG = bbox[2]
    UP_BORDER_TRG = bbox[3]

    cell_size_hor = (RIGHT_BORDER - LEFT_BORDER) / n_cells_hor
    cell_size_ver = (UP_BORDER - DOWN_BORDER) / n_cells_ver

    celled_data_trg = np.zeros([n_cells_hor, n_cells_ver])
    cell_size_hor_trg = (RIGHT_BORDER_TRG - LEFT_BORDER_TRG) / n_cells_hor
    cell_size_ver_trg = (UP_BORDER_TRG - DOWN_BORDER_TRG) / n_cells_ver

    x_arr = np.zeros(celled_data.shape[0])
    y_arr = np.zeros(celled_data.shape[1])
    x_y_arr = []
    for i in range(celled_data.shape[0]):
        for j in range(celled_data.shape[1]):
            if celled_data[i, j] > 0:
                x_src = float(i) * float(cell_size_hor) + LEFT_BORDER
                y_src = float(j) * float(cell_size_ver) + DOWN_BORDER
                lon, lat = cartesian_to_spherical(x_src, y_src)
                x_y_arr.append((lon, lat))

    x_y_arr = np.asarray(x_y_arr)
    if len(x_y_arr) > 0:
        return np.asarray(x_y_arr)[:, 0], np.asarray(x_y_arr)[:, 1]
    else:
        return None


def get_coords_map(
    celled_data, n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, bbox=BBOX
):
    LEFT_BORDER = 0
    RIGHT_BORDER = 2000
    DOWN_BORDER = 0
    UP_BORDER = 2500

    LEFT_BORDER_TRG = bbox[0]
    RIGHT_BORDER_TRG = bbox[1]
    DOWN_BORDER_TRG = bbox[2]
    UP_BORDER_TRG = bbox[3]

    cell_size_hor = (RIGHT_BORDER - LEFT_BORDER) / n_cells_hor
    cell_size_ver = (UP_BORDER - DOWN_BORDER) / n_cells_ver

    n = N_CELLS_HOR
    celled_data_trg = np.zeros([n, n])
    cell_size_hor_trg = (RIGHT_BORDER_TRG - LEFT_BORDER_TRG) / n
    cell_size_ver_trg = (UP_BORDER_TRG - DOWN_BORDER_TRG) / n

    x_arr = np.zeros(celled_data.shape[0])
    y_arr = np.zeros(celled_data.shape[1])
    for i in range(celled_data.shape[0]):
        for j in range(celled_data.shape[1]):
            x_src = float(i) * float(cell_size_hor) + LEFT_BORDER
            y_src = float(j) * float(cell_size_ver) + DOWN_BORDER
            lon, lat = cartesian_to_spherical(x_src, y_src)
            x_arr[i] = lon
            y_arr[j] = lat

            if (
                (lon > LEFT_BORDER_TRG)
                & (lon < RIGHT_BORDER_TRG)
                & (lat > DOWN_BORDER_TRG)
                & (lat < UP_BORDER_TRG)
            ):
                x_trg = int(((lon - LEFT_BORDER_TRG) / cell_size_hor_trg))
                y_trg = int(((lat - DOWN_BORDER_TRG) / cell_size_ver_trg))
                celled_data_trg[x_trg, y_trg] = celled_data[i, j]

    return celled_data_trg, x_arr, y_arr


def plot_target_pred(
    target_np,
    prediction_np,
    trs=0.0,
    save=True,
    savepath="target_pred.png",
    dpi=DPI,
    text=None,
):
    prediction_np[prediction_np < trs] = 0.0
    prediction_unproj, _, _ = get_coords_map(prediction_np)
    plot_density(prediction_unproj.transpose()[::-1])

    coords = get_coords(target_np)
    if coords is not None:
        plt.scatter(coords[0], coords[1], marker="x", c="k", s=100)

    if text is not None:
        plt.title(text)

    plt.xlim((127.0, 147.0))
    plt.ylim((27.0, 45.98))

    if save:
        plt.savefig(savepath, dpi=dpi)
        plt.close()
