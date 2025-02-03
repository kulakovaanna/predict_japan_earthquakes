from copy import deepcopy

import numpy as np
from tqdm import tqdm
from numpy.typing import NDArray
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc, precision_recall_fscore_support, 
    average_precision_score, RocCurveDisplay, confusion_matrix,
)
import matplotlib.pyplot as plt

from utils import (
    preprocess_time,
    get_cell_pos_and_center,
)


DATA_ORIG_PATH = "../data/catalogues/originalCat.csv"
DATA_WITHOUT_AFT_PATH = "../data/catalogues/withoutAftCat.csv"
glob_params_dict = {
    "magn_6_aft": {
        "DF_FOR_FEATURES": DATA_ORIG_PATH,
        "min_magnitude": 6.0,
        "trs": 6.0,
        "window_days_step": 1,
    },
    "magn_6_withoutaft": {
        "DF_FOR_FEATURES": DATA_WITHOUT_AFT_PATH,
        "min_magnitude": 6.0,
        "trs": 6.0,
        "window_days_step": 1,
    },
    "magn_6_aft_test": {
        "DF_FOR_FEATURES": DATA_ORIG_PATH,
        "min_magnitude": 6.0,
        "trs": 6.0,
        "window_days_step": 1,
    },
    "magn_6_withoutaft_test": {
        "DF_FOR_FEATURES": DATA_WITHOUT_AFT_PATH,
        "min_magnitude": 6.0,
        "trs": 6.0,
        "window_days_step": 1,
    },
    "magn_3_5_aft_train": {
        "DF_FOR_FEATURES": DATA_ORIG_PATH,
        "min_magnitude": 3.5,
        "trs": 3.5,
        "window_days_step": 10,
    },
    "magn_3_5_withoutaft_train": {
        "DF_FOR_FEATURES": DATA_WITHOUT_AFT_PATH,
        "min_magnitude": 3.5,
        "trs": 3.5,
        "window_days_step": 10,
    },
    "magn_3_5_aft_test": {
        "DF_FOR_FEATURES": DATA_ORIG_PATH,
        "min_magnitude": 3.5,
        "trs": 3.5,
        "window_days_step": 1,
    },
    "magn_3_5_withoutaft_test": {
        "DF_FOR_FEATURES": DATA_WITHOUT_AFT_PATH,
        "min_magnitude": 3.5,
        "trs": 3.5,
        "window_days_step": 1,
    },
}


def get_features_dict(cols: list[str]) -> dict[int, list[int]]:
    def _get_feature_name_depth(feature_name: str) -> int:
        if "earthquakes_" in feature_name:
            return int(feature_name.split("earthquakes_")[-1].split("_")[0])//365
        elif "magn_" in feature_name:
            return int(feature_name.split("magn_")[-1].split("_")[0])//365
    
    return {
        r:np.sort(list(set([
            _get_feature_name_depth(x)
            for x in cols if ("earthquakes" in x or "magn" in x) and f"rad{r}" in x
        ]))).tolist()
        for r in [0, 3, 5]
    }
    
def get_train_test_datasets(
    train_path: str, test_path: str, 
    target: str = "target", dt_col: str = "dt", 
    min_test_date: str = "2020-10-14", min_train_date: str = "2014-03-17"
) -> tuple[pd.DataFrame, pd.DataFrame]:

    print(f"train_path: {train_path}")
    print(f"test_path: {test_path}\n")
    
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    intersection_cols = set(train_df.columns).intersection(set(test_df.columns))
    train_drop_cols = [x for x in train_df.columns if x not in intersection_cols]
    test_drop_cols = [x for x in test_df.columns if x not in intersection_cols]
    
    train_df = train_df.drop(columns=train_drop_cols)
    test_df = test_df.drop(columns=test_drop_cols)

    train_df = train_df.query(f"dt < '{min_test_date}'").query(f"dt >= '{min_train_date}'")
    test_df = test_df.query(f"dt >= '{min_test_date}'")
        
    assert set(train_df.columns) == set(test_df.columns), f"{set(train_df.columns)}, {set(test_df.columns)}"
    
    print("get_features_dict:\n", str(get_features_dict(train_df.columns)).replace("], ", "]\n"))
    print(f"\ntrain dates: {np.sort(train_df.dt.unique())[0]} — {np.sort(train_df.dt.unique())[-1]}")
    print(f"test dates: {np.sort(test_df.dt.unique())[0]} — {np.sort(test_df.dt.unique())[-1]}")
    print(f"\ntrain: target==0: {len(train_df[train_df[target]==0])}, target==1: {len(train_df[train_df[target]==1])}")
    print(f"test: target==0: {len(test_df[test_df[target]==0])}, target==1: {len(test_df[test_df[target]==1])}")
    print(f"target class proportion: test: {len(test_df[test_df[target]==1])/len(test_df[test_df[target]==0])}, train: {len(train_df[train_df[target]==1])/len(train_df[train_df[target]==0])}")

    return train_df, test_df


def get_weights_for_roc_auc(y: pd.DataFrame, last_dt: str, trs: float = 3.0, norm: bool = True, fillna: bool = True):        
    df = pd.read_csv(DATA_ORIG_PATH, sep=" ")
    df = df[df["Class"] > trs].reset_index(drop=True)
    df["time"] = df[["YYYY", "MM", "DD", "HH", "mm", "ssss"]].apply(lambda x: preprocess_time(*x), axis=1)
    df = df.rename(
        columns={
            "LAT": "lat",
            "LONG": "lon",
            "Class": "magn",
            "Depth": "depth",
            "YYYY": "year",
            "MM": "month",
            "DD": "day",
            "HH": "hour",
            "mm": "minute",
            "ssss": "second",
        }
    )
    df = df.rename(columns={"longitude": "lon", "latitude": "lat", "magnitude": "magn"})
    df["time"] = df["time"].apply(lambda x: x.replace("Z", ""))
    df["time"] = df["time"].astype("datetime64[ns]")
    df["dt"] = df["time"].apply(lambda x: x.strftime("%Y-%m-%d"))
    df = df.query(f"dt < 'last_dt'").reset_index(drop=True)
    df["cell_x"] = df[["lon", "lat"]].apply(lambda x: get_cell_pos_and_center(lon=x[0], lat=x[1])[0], axis=1)
    df["cell_y"] = df[["lon", "lat"]].apply(lambda x: get_cell_pos_and_center(lon=x[0], lat=x[1])[1], axis=1)
    df["event"] = 1.0

    density_df = (
        df
        [["cell_x", "cell_y", "event"]]
        .groupby(by=["cell_x", "cell_y"]).sum("event")
        .rename(columns={"event": "freq"})
        .reset_index()
    )
    if norm:
        max_events = density_df["freq"].max()
        density_df["freq"] = density_df["freq"] / max_events
        
    weights = y.reset_index().merge(
        density_df[["cell_x", "cell_y", "freq"]],
        on=["cell_x", "cell_y"],
        how="left"
    )["freq"]
    
    if fillna:
        weights = weights.fillna(0.)
        
    return weights.to_numpy()

def get_tpr_fpr(y_true, y_pred, sample_weight=None, n=25):
    tpr, fpr, fpr_w = [], [], []
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

def plot_roc_curves(y_true, y_score, sample_weight, n=25, title=""):
    tpr, fpr = get_tpr_fpr(y_true, y_score, sample_weight=None, n=n)
    _, fpr_w = get_tpr_fpr(y_true, y_score, sample_weight=sample_weight, n=n)
    roc_auc = np.around(auc(fpr, tpr), 4)
    roc_auc_w = np.around(auc(fpr_w, tpr), 4)
    
    fig, ax = plt.subplots(figsize=(6, 6), dpi=100)
    ax.plot(range(2), range(2), 'grey', ls="--")
    ax.plot(fpr, tpr, c="orange", label=f"roc auc = {roc_auc}")
    ax.plot(fpr_w, tpr, c="blue", label=f"roc auc weighted = {roc_auc_w}")
    plt.grid(alpha=0.4)
    plt.legend()
    plt.title(title)
    plt.show()
    
def weighted_roc_metric(y, predict, min_test_date="2020-10-14", target="target"):
    if isinstance(y, pd.Series):
        y = pd.DataFrame(y)
    # last_dt = str(np.sort(y.reset_index()["dt"].unique())[-1])
    weights_for_roc_auc = get_weights_for_roc_auc(y=y, last_dt=min_test_date)
    
    return roc_auc_score(y[target], predict, sample_weight=weights_for_roc_auc)

def weighted_fpr_roc_metric(y, predict, min_test_date="2020-10-14", target="target"):
    if isinstance(y, pd.Series):
        y = pd.DataFrame(y)
    # last_dt = str(np.sort(y.reset_index()["dt"].unique())[-1])
    weights_for_roc_auc = get_weights_for_roc_auc(y=y, last_dt=min_test_date)

    tpr, fpr = get_tpr_fpr(y[target], predict, sample_weight=weights_for_roc_auc)
    roc_auc_score = auc(fpr, tpr)
    
    return roc_auc_score

def weighted_roc_metric_harder_magn(y, predict, min_test_date="2020-10-14", target="target"):
    if isinstance(y, pd.Series):
        y = pd.DataFrame(y)
    # last_dt = str(np.sort(y.reset_index()["dt"].unique())[-1])
    weights_for_roc_auc = get_weights_for_roc_auc(y=y, last_dt=min_test_date, trs=5.0)

    # tpr, fpr = get_tpr_fpr(y[target], predict, sample_weight=weights_for_roc_auc)
    # roc_auc_score = auc(fpr, tpr)
    
    return roc_auc_score(y[target], predict, sample_weight=weights_for_roc_auc)

def get_optimal_trs(target, predicted, sample_weight):
    fpr, tpr, threshold = roc_curve(target, predicted, sample_weight=sample_weight)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def plot_roc(target, prediction, weights=None, src_label="", save_dir=None, file_name=None):   
    assert prediction.shape == target.shape 
    ROC_AUC_score = np.around(roc_auc_score(y_true=target, y_score=prediction, sample_weight=weights), 4)

    metrics_str = f"ROC AUC = {ROC_AUC_score}"
    if isinstance(src_label, str) and len(src_label)>0:
        label = src_label+f" ROC AUC ={ROC_AUC_score}"
    else:
        label = metrics_str
    
    fpr, tpr, _ = roc_curve(y_true=target, y_score=prediction, sample_weight=weights)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)

    fig, ax1 = plt.subplots(figsize=(7, 7), dpi=80)
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
    
def get_df_for_agg(df_path: str, sep: str = " ", min_magn: float = 3.5):
    df = pd.read_csv(df_path, sep=sep)
    df = df[df["Class"] > min_magn].reset_index(drop=True)
    df["time"] = df[["YYYY", "MM", "DD", "HH", "mm", "ssss"]].apply(lambda x: preprocess_time(*x), axis=1)
    df = df.rename(
        columns={
            "LAT": "lat",
            "LONG": "lon",
            "Class": "magn",
            "Depth": "depth",
            "YYYY": "year",
            "MM": "month",
            "DD": "day",
            "HH": "hour",
            "mm": "minute",
            "ssss": "second",
        }
    )
    df = df.rename(columns={"longitude": "lon", "latitude": "lat", "magnitude": "magn"})
    df["time"] = df["time"].apply(lambda x: x.replace("Z", ""))
    df["time"] = df["time"].astype("datetime64[ns]")
    df["cell_x"] = df[["lon", "lat"]].apply(lambda x: get_cell_pos_and_center(lon=x[0], lat=x[1])[0], axis=1)
    df["cell_y"] = df[["lon", "lat"]].apply(lambda x: get_cell_pos_and_center(lon=x[0], lat=x[1])[1], axis=1)

    return df

class Features:
    def __init__(
        self,
        *,
        features: list[str] | None = None,
        target: str = None,
        target_features: list[str] | None = None,
        groupby: list[str] | None = None,
        dt_col: str = None,
    ) -> None:
        if groupby is None:
            groupby = ["name"]
        if target is None:
            target = "d0"
        if features is None:
            features = pd.read_csv(f"data/use_cols_{target}.csv").columns
        if target_features is None:
            if target == "d":
                target_features = ["month_date"]
            else:
                target_features = ["opening_date"]
        if dt_col is None:
            if target == "d":
                dt_col = "month_date"
            else:
                dt_col = "opening_date"
        target_features += [target]

        self.features = deepcopy(features)
        self.target = deepcopy(target)
        self.target_features = deepcopy(target_features)
        self.dt_col = dt_col
        self.groupby = groupby

    def make_features_target(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        if len(self.groupby) > 0:
            features_df = df.set_index([self.dt_col, *self.groupby], drop=False)[self.features].copy()
            target_df = df.set_index([self.dt_col, *self.groupby], drop=False)[self.target_features].copy()
        else:
            features_df = df.set_index([self.dt_col], drop=False)[self.features].copy()
            target_df = df.set_index([self.dt_col], drop=False)[self.target_features].copy()
        return features_df, target_df

    def numpy_to_target(self, X: NDArray) -> pd.DataFrame:
        target_df = pd.DataFrame(X, columns=self.target_features, copy=True)
        return target_df


def get_dt_from_index(X: pd.DataFrame | pd.Series | pd.Index, format: str = "%Y-%m-%d") -> pd.DatetimeIndex:
    if isinstance(X, pd.Index):
        I = X
    else:
        I = X.index
    for l in range(I.nlevels):
        idx = I.get_level_values(l)
        dt = pd.to_datetime(idx, format=format, errors="coerce")
        if not dt.hasnans:
            return dt
    raise Exception("No date level found in the index.")


class MonthlyTimeSeriesSplit:
    def __init__(
        self,
        *,
        window: int = 1,
        gap: int = 0,
        min_train_size: int | None = None,
        min_test_size: int = 1,
        skip_empty: bool = True,
        partition: bool = False,
    ) -> None:
        if min_train_size is None:
            min_train_size = window
        self.window = window
        self.gap = gap
        self.min_train_size = min_train_size
        self.min_test_size = min_test_size
        self.skip_empty = skip_empty
        self.partition = partition

    def _get_t(self, X) -> pd.Series:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Pandas DataFrame is required.")
        dt = get_dt_from_index(X)
        t = dt.year * 12 + dt.month
        t -= t.min()
        return t.to_numpy()

    def split(self, X, y=None, groups=None):
        t = self._get_t(X)
        for i in range(self._get_n_splits(X)):
            train_size = self.min_train_size + i * self.window
            train = np.arange(t.size)[t < train_size]
            test = np.arange(t.size)[(train_size + self.gap <= t) & (t < train_size + self.gap + self.window)]
            if not self.skip_empty or test.size >= self.min_test_size:
                yield train, test
        if self.partition:
            # fictitious split so that all tests form a partition of the dataset
            train = np.arange(t.size)[t < self.min_train_size + self.gap]
            test = train.copy()
            yield train, test

    def _get_n_splits(self, X, y=None, groups=None):
        t = self._get_t(X)
        n_months = t.max() + 1
        total_test_months = max(0, n_months - self.min_train_size - self.gap)
        n_splits = (total_test_months + self.window - 1) // self.window
        return n_splits

    def get_n_splits(self, X, y=None, groups=None):
        return sum(1 for _ in self.split(X))