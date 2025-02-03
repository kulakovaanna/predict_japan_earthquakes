import os
import math as m
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import roc_auc_score

from train import *
from model import *
from config import *


def my_precision_TPR_FPR(input, target, threshold):
    TP = torch.sum(((input > threshold).float() * target).float())
    FP = torch.sum(((input > threshold).float() * (1 - target)).float())
    FN = torch.sum(((~(input > threshold)).float() * target).float())
    TN = torch.sum(((~(input > threshold)).float() * (1 - target)).float())

    return TP / (TP + FP), TP / (TP + FN), FP / (FP + TN)


def check_quality(RNN_cell, device, dataloader_test, n_dots=501, info_file=None):

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
    for data in tqdm(dataloader_test):

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

    print("ROC_AUC = ", end="")

    trg = np.array(target.view(-1).cpu())
    prd = np.array(prediction.view(-1).cpu())

    ROC_AUC_score = roc_auc_score(
        np.array(target.view(-1).cpu()), np.array(prediction.view(-1).cpu())
    )
    print(ROC_AUC_score)

    threshold_massive = torch.linspace(0, 1, n_dots, dtype=torch.float, device=device)

    precision_massive = []
    recall_massive = []
    FPR_massive = []

    for threshold in tqdm(threshold_massive):
        precision, recall, FPR = my_precision_TPR_FPR(prediction, target, threshold)
        precision_massive.append(precision.item())
        recall_massive.append(recall.item())
        FPR_massive.append(FPR.item())

    fig = plt.figure(figsize=(8, 8))

    axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])

    #     axes.plot(FPR_massive, recall_massive, 'blue', marker = '.')
    axes.plot(FPR_massive, recall_massive, "blue")
    axes.plot(range(2), range(2), "grey", ls="--")
    axes.grid(alpha=0.4)

    axes.set_xlabel("False Positive Rate")
    axes.set_ylabel("True Positive Rate")
    axes.set_title(f"roc auc = {np.around(ROC_AUC_score, 4)}")

    plt.show()

    return ROC_AUC_score


def get_target_pred(RNN_cell, device, dataloader_test, n_dots=501, info_file=None):

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
    for data in tqdm(dataloader_test):

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
        batch_size=1,
        shuffle=False,
        num_workers=10,
    ):

        self.celled_data_x_path = celled_data_x_path
        self.celled_data_y_path = celled_data_y_path
        self.celled_data_path_for_freq_map = celled_data_path_for_freq_map
        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver
        self.model_name = model_name
        self.testing_days = testing_days
        self.heavy_quake_thres = heavy_quake_thres
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
        self.device = device
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

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
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def save_model(self):
        if not os.path.exists("../data/model"):
            os.mkdir("../data/model")
        torch.save(self.model.state_dict(), f"../data/model/{self.model_name}")

    def init_model(self):
        if self.two_maps_flag == False:
            self.model = LSTMCell(
                frequency_map=self.freq_map,
                embedding_size=self.embedding_size,
                hidden_state_size=self.hidden_state_size,
                n_cells_hor=self.n_cells_hor,
                n_cells_ver=self.n_cells_ver,
                device=self.device,
            )

        elif self.two_maps_flag == True:
            self.model = LSTMCell_density(
                frequency_map=self.freq_map,
                embedding_size=self.embedding_size,
                hidden_state_size=self.hidden_state_size,
                n_cells_hor=self.n_cells_hor,
                n_cells_ver=self.n_cells_ver,
                device=self.device,
            )

    def train(self):
        self.model.train()
        train_RNN_full(
            self.model,
            self.device,
            self.dataloader_train,
            n_cycles=self.n_cycles,
            learning_rate=self.learning_rate,
            earthquake_weight=self.earthquake_weight,
            lr_decay=self.lr_decay,
            start_lr_decay=self.start_lr_decay,
            weight_decay=self.weight_decay,
        )

        self.save_model()

    def validate(self):
        self.model.eval()
        ROC_AUC = check_quality(
            self.model, self.device, self.dataloader_test, n_dots=251
        )

    def __call__(self):
        print("initializing datasets, dataloaders, model . . . ")
        self.get_celled_data_x_y()
        self.get_freq_map()
        print(f"two_maps_flag: {self.two_maps_flag}")
        self.get_train_test_datasets()
        self.get_train_test_dataloaders()
        self.init_model()

        print("\ntraining . . .")
        self.train()

        print("\nvalidating . . .")
        self.validate()


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
