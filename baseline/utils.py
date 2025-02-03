import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, confusion_matrix,
    average_precision_score, precision_recall_curve,
    RocCurveDisplay, PrecisionRecallDisplay
)
from tqdm import tqdm

from constants import MAP_PATH, FIGSIZE, BBOX, N_CELLS_HOR, N_CELLS_VER, DPI, DATA_ORIG_PATH


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

def create_celled_data(lon, lat, n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, bbox=BBOX, density=True):
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
        
    celled_data = celled_data.transpose()[::-1]
    
    x_arr = np.zeros(celled_data.shape[0])
    y_arr = np.zeros(celled_data.shape[1])
    for i in range(celled_data.shape[0]):
        for j in range(celled_data.shape[1]):
            x_arr[i] = i*cell_size_hor + LEFT_BORDER
            y_arr[j] = j*cell_size_ver + DOWN_BORDER

    return celled_data, x_arr, y_arr

def check_quality(target, prediction, src_label="", save_dir=None, file_name=None):   
    assert prediction.shape == target.shape 
    ROC_AUC_score = np.around(roc_auc_score(y_true=target, y_score=prediction), 4)
    AVG_precision_score = np.around(average_precision_score(y_true=target, y_score=prediction), 4)

    metrics_str = f"roc_auc={ROC_AUC_score}, avg_precision={AVG_precision_score}"
    if isinstance(src_label, str) and len(src_label)>0:
        label = src_label+f", roc_auc={ROC_AUC_score}, avg_precision={AVG_precision_score}"
    else:
        label = metrics_str
    
    fpr, tpr, _ = roc_curve(y_true=target, y_score=prediction)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    precision, recall, _ = precision_recall_curve(target, prediction)
    pr_display = PrecisionRecallDisplay(precision=precision, recall=recall)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=DPI)
    roc_display.plot(ax=ax1, c="b")
    pr_display.plot(ax=ax2, c="b")
    fig.suptitle(label)
    ax1.grid(alpha=0.4)
    ax2.grid(alpha=0.4)

    if save_dir is None:
        plt.show()
    else:
        if file_name is None:
            if src_label == "":
                file_name = label.replace(", ", "_").replace(" ", "_").replace(".", "_")
            else:
                file_name = src_label.replace(", ", "_").replace(" ", "_").replace(".", "_")
        else:
            file_name = file_name + "_" + src_label.replace(", ", "_").replace(" ", "_").replace(".", "_").replace("<", "").replace("=", "")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{file_name}.png"), dpi=150)

    return ROC_AUC_score, AVG_precision_score


def plot_roc(target, prediction, src_label="", save_dir=None, file_name=None):   
    assert prediction.shape == target.shape 
    ROC_AUC_score = np.around(roc_auc_score(y_true=target, y_score=prediction), 4)

    metrics_str = f"ROC AUC = {ROC_AUC_score}"
    if isinstance(src_label, str) and len(src_label)>0:
        label = src_label+f", ROC AUC ={ROC_AUC_score}"
    else:
        label = metrics_str
    
    fpr, tpr, _ = roc_curve(y_true=target, y_score=prediction)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    fig, ax1 = plt.subplots(figsize=(7, 6), dpi=DPI)
    roc_display.plot(ax=ax1, c="b")
    ax1.plot (range(2), range(2), 'grey', ls='--')
    fig.suptitle(label)
    ax1.grid(alpha=0.4)
    ax1.legend().set_visible(False)

    if save_dir is None:
        plt.show()
    else:
        if file_name is None:
            if src_label == "":
                file_name = label.replace(", ", "_").replace(" ", "_").replace(".", "_")
            else:
                file_name = src_label.replace(", ", "_").replace(" ", "_").replace(".", "_")
        else:
            file_name = file_name + "_" + src_label.replace(", ", "_").replace(" ", "_").replace(".", "_").replace("<", "").replace("=", "")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{file_name}.png"), dpi=150)

    return ROC_AUC_score


def calc_rocauc_avgprec(target, prediction):   
    assert prediction.shape == target.shape 
    ROC_AUC_score = np.around(roc_auc_score(y_true=target, y_score=prediction), 4)
    AVG_precision_score = np.around(average_precision_score(y_true=target, y_score=prediction), 4)
    
    return ROC_AUC_score, AVG_precision_score

def plot_density(density_map, map_path=MAP_PATH, figsize=FIGSIZE, bbox=BBOX):
    map_img = plt.imread(map_path)
    fig, ax = plt.subplots(figsize=(figsize, figsize * map_img.shape[0] / map_img.shape[1]), dpi=DPI)
    ax.imshow(map_img, zorder=0, extent=bbox);
    plt.imshow(density_map, alpha=0.7, interpolation='bilinear', extent=bbox, cmap="seismic");
    
def plot_events(x, y, map_path=MAP_PATH, figsize=FIGSIZE, bbox=BBOX):
    map_img = plt.imread(map_path)
    fig, ax = plt.subplots(figsize=(figsize, figsize * map_img.shape[0] / map_img.shape[1]), dpi=DPI)
    ax.imshow(map_img, zorder=0, extent=bbox);
    plt.scatter(x, y, marker="x", c="k");

def get_tpr_fpr(y_true, y_pred, sample_weight=None, n=25):
    tpr, fpr = [], []
    full_trs_list_ = [-0.001] + np.unique(y_pred).tolist() + [1.001]
    trs_list_ = np.linspace(-0.001, 1.001, n)
    trs_list = trs_list_ if len(trs_list_) < len(full_trs_list_) else full_trs_list_

    for trs in tqdm(trs_list, leave=False):
        y_pred_ = np.array(y_pred >= trs, dtype=int)
        cm = confusion_matrix(y_true, y_pred_, sample_weight=None)
        if sample_weight is not None:
            cm_w = confusion_matrix(y_true, y_pred_, sample_weight=sample_weight)
    
        if sample_weight is None:
            tn, fp, fn, tp = cm.ravel()
        else:
            _, _, fn, tp = cm.ravel()
            tn, fp, _, _ = cm_w.ravel()
            
        tpr_ = tp / (tp+fn)
        fpr_ = fp / (fp+tn)
        tpr.append(tpr_)
        fpr.append(fpr_)

    return tpr, fpr

def weighted_fpr_roc(y_true, y_pred, weights_for_roc_auc, n=25):
    tpr, fpr = get_tpr_fpr(y_true, y_pred, sample_weight=weights_for_roc_auc, n=n)
    roc_auc_score = auc(fpr, tpr)
    
    return roc_auc_score

def weighted_roc_curve(y_true, y_score, sample_weight, n=25, title=""):
    tpr, fpr_w = get_tpr_fpr(y_true, y_score, sample_weight=sample_weight, n=n)
    roc_auc_w = np.around(auc(fpr_w, tpr), 4)
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.plot(range(2), range(2), 'grey', ls="--")
    ax.plot(fpr_w, tpr, c="blue", label=f"roc auc weighted = {roc_auc_w}")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.title(title)
    plt.show()