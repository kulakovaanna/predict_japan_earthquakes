import warnings
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from utils import (
    get_cell_pos_and_center,
)
from ml_utils import glob_params_dict, get_df_for_agg, DATA_ORIG_PATH

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


def get_rad(x, y, radius):
    if radius == 0:
        x_rad = [x]
        y_rad = [y]
    else:
        x_rad = list(range(x - radius, x + radius))
        y_rad = list(range(y - radius, y + radius))

    return x_rad, y_rad


depth_list = [
    60,
    90,
    180,
    365,
    # 3 * 365,
    # 5 * 365,
    # 7 * 365,
    # 10 * 365,
]
rad_list = [
    0, 
    # 3, 
    # 5
]

dataset_type = "magn_3_5_withoutaft_test"
params_dict = glob_params_dict[dataset_type]
depth_str = "_".join([str(x) for x in depth_list])
rad_str = "_".join([str(x) for x in rad_list])

DATASET_SAVE_PATH = f"data/features/{dataset_type}_features_3_5_rad{rad_str}_depth{depth_str}_trs{params_dict['trs']}.parquet"
DF_FOR_FEATURES = params_dict["DF_FOR_FEATURES"]
min_magnitude = params_dict["min_magnitude"]
trs = params_dict["trs"]
window_days_step = params_dict["window_days_step"]
features_trs_list = params_dict["features_trs"]

print("params_dict:", params_dict)
print()


# forming cells
print("DATA_ORIG_PATH", DATA_ORIG_PATH)
orig_df = pd.read_csv(DATA_ORIG_PATH, sep=" ")
print("len(orig_df)", len(orig_df))

target_cells_df = get_df_for_agg(df_path=DF_FOR_FEATURES, min_magn=trs)[["cell_x", "cell_y", "time"]]
target_cells_df = target_cells_df.drop_duplicates().reset_index(drop=True)
features_df = get_df_for_agg(df_path=DF_FOR_FEATURES, min_magn=min_magnitude)

# get dates for dataset
horizon_min = 10
horizon_max = 50
dataset_date_cols = ["dt", "horizon_min_dt", "horizon_max_dt"]
window_dt = datetime.timedelta(days=window_days_step)
horizon_min_dt = datetime.timedelta(days=horizon_min)
horizon_max_dt = datetime.timedelta(days=horizon_max)

if "test" in dataset_type:
    start_dt = features_df["time"].max() - relativedelta(days=1000)
    end_dt = features_df["time"].max()
else:
    start_dt = features_df["time"].min() + relativedelta(years=5)  # for a feature using history of last 5 years
    end_dt = features_df["time"].max() - relativedelta(days=1000)


current_dt = start_dt
dates = []

while current_dt < (end_dt - horizon_max_dt):
    current_dt += window_dt
    dates.append(str(current_dt.date()))

print(f"dates: {len(dates)}")


# create empty dataset
dataset = orig_df[["LAT", "LONG", "Class"]].rename(columns={"LAT": "lat_src", "LONG": "lon_src"}).copy()
print("min_magnitude", min_magnitude)
dataset = (
    dataset[dataset["Class"] > min_magnitude].drop(columns=["Class"]).drop_duplicates().reset_index(drop=True)
)
print("len(dataset) > min_magn", len(dataset))
dataset["cell_x"] = dataset[["lon_src", "lat_src"]].apply(
    lambda x: get_cell_pos_and_center(lon=x[0], lat=x[1])[0], axis=1
)
dataset["cell_y"] = dataset[["lon_src", "lat_src"]].apply(
    lambda x: get_cell_pos_and_center(lon=x[0], lat=x[1])[1], axis=1
)
dataset["lon_cell"] = dataset[["lon_src", "lat_src"]].apply(
    lambda x: get_cell_pos_and_center(lon=x[0], lat=x[1])[2], axis=1
)
dataset["lat_cell"] = dataset[["lon_src", "lat_src"]].apply(
    lambda x: get_cell_pos_and_center(lon=x[0], lat=x[1])[3], axis=1
)
dataset = dataset.drop(columns=["lon_src", "lat_src"]).drop_duplicates().reset_index(drop=True)
dataset["dt"] = pd.Series([dates] * len(dataset))
dataset = dataset.explode("dt")
dataset = dataset[["dt", "lon_cell", "lat_cell", "cell_x", "cell_y"]]
dataset["horizon_min_dt"] = dataset["dt"].apply(
    lambda x: datetime.date(int(x.split("-")[0]), int(x.split("-")[1]), int(x.split("-")[2])) + horizon_min_dt
)
dataset["horizon_max_dt"] = dataset["dt"].apply(
    lambda x: datetime.date(int(x.split("-")[0]), int(x.split("-")[1]), int(x.split("-")[2])) + horizon_max_dt
)

dataset = dataset.drop_duplicates().reset_index(drop=True)
dataset[dataset_date_cols] = dataset[dataset_date_cols].astype("datetime64[ns]")

print(f"dataset size: {len(dataset)}")


feature_list = ["count_earthquakes", "mean_magn", "min_magn", "max_magn", "std_magn"]

print()
print("depth_list:", depth_list)
print("rad_list:", rad_list)

dataset["target"] = 0
# for depth in depth_list:
#     for rad in rad_list:
#         feature_depth_list = [f"{x}_{depth}_rad{rad}" for x in feature_list]
#         for feature_name in feature_depth_list:
#             dataset[feature_name] = 0.0

horizon_target_min = 10
horizon_target_max = 50
horizon_target_min_dt = datetime.timedelta(days=horizon_target_min)
horizon_target_max_dt = datetime.timedelta(days=horizon_target_max)

horizon_min_dt = datetime.timedelta(days=1)
for i, row in tqdm(dataset.iterrows(), total=len(dataset), mininterval=120.0):
    x = row["cell_x"]
    y = row["cell_y"]
    date = row["dt"].date()

    # target
    horizon_target_cell_df = target_cells_df[
        (target_cells_df["cell_x"] == x)
        & (target_cells_df["cell_y"] == y)
        & (target_cells_df["time"] >= np.datetime64(date + horizon_target_min_dt))
        & (target_cells_df["time"] <= np.datetime64(date + horizon_target_max_dt))
    ]
    if len(horizon_target_cell_df) > 0:
        dataset.loc[i, "target"] = 1

    # features
    for depth in depth_list:
        horizon_max_dt = datetime.timedelta(days=depth)

        for rad in rad_list:
            x_rad, y_rad = get_rad(x, y, rad)

            for features_trs in features_trs_list:
                horizon_magn_cell = features_df[
                    (features_df["cell_x"].isin(x_rad))
                    & (features_df["cell_y"].isin(y_rad))
                    & (features_df["time"] < np.datetime64(date - horizon_min_dt))
                    & (features_df["time"] >= np.datetime64(date - horizon_max_dt))
                    & (features_df["magn"] > features_trs)
                ]["magn"].to_numpy()
                value = len(horizon_magn_cell)
    
                if value > 0:
                    dataset.loc[i, f"count_earthquakes_m{features_trs}_{depth}_rad{rad}"] = value
                    dataset.loc[i, f"mean_magn_m{features_trs}_{depth}_rad{rad}"] = np.mean(horizon_magn_cell)
                    dataset.loc[i, f"min_magn_m{features_trs}_{depth}_rad{rad}"] = np.min(horizon_magn_cell)
                    dataset.loc[i, f"max_magn_m{features_trs}_{depth}_rad{rad}"] = np.max(horizon_magn_cell)
                    dataset.loc[i, f"std_magn_m{features_trs}_{depth}_rad{rad}"] = np.std(horizon_magn_cell)

dataset.to_parquet(DATASET_SAVE_PATH, index=False)
