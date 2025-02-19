DAYS_TO_PREDICT_AFTER = 10
DAYS_TO_PREDICT_BEFORE = 50
TESTING_DAYS = 1000
EMB_SIZE = 16
HID_SIZE = 32

NUM_WORKERS = 16
DEVICE = "cuda:0"

ORIGIN_LATITUDE = 27.0
ORIGIN_LONGITUDE = 127.0
EARTH_RADIUS = 6373.0

N_CELLS_HOR = 200
N_CELLS_VER = 250
BBOX = (123.43, 149.18, 25.41, 45.98)

BBOX_SMALLER = (127.0, 147.0, 27.5, 45.98)
MAP_PATH = "data/jp_map.png"
FIGSIZE = 10
DPI = 80

DATA_ORIG_PATH = "../data/catalogs/originalCat.csv"


# from orinal paper
CELLED_DATA_PATH_CAT_ORIG_SRC = "../data/celled_data_cat_orig"
CELLED_DATA_PATH_WITHOUTAFT_CAT_SRC = "../data/celled_data_without_aft_cat"
# density map
CELLED_DATA_PATH_CAT_ORIG_SRC_DENSITY = "../data/celled_data_cat_orig_src_density"
CELLED_DATA_PATH_WITHOUTAFT_CAT_SRC_DENSITY = (
    "../data/celled_data_without_aft_catsrc_density"
)
# max magnitude map
CELLED_DATA_PATH_CAT_ORIG_MAXMAGN = "../data/celled_data_cat_orig_maxmagn"
CELLED_DATA_PATH_WITHOUTAFT_CAT_MAXMAGN = "../data/celled_data_without_aft_cat_maxmagn"
# density and max magnitude map
CELLED_DATA_PATH_CAT_ORIG_MAXMAGN_DENSITY = (
    "../data/celled_data_cat_orig_maxmagn_density"
)
CELLED_DATA_PATH_WITHOUTAFT_CAT_MAXMAGN_DENSITY = (
    "../data/celled_data_without_aft_cat_maxmagn_density"
)


default_params = {
    "n_cells_hor": N_CELLS_HOR,
    "n_cells_ver": N_CELLS_VER,
    "testing_days": TESTING_DAYS,
    "days_to_predict_before": DAYS_TO_PREDICT_BEFORE,
    "days_to_predict_after": DAYS_TO_PREDICT_AFTER,
    "embedding_size": EMB_SIZE,
    "hidden_state_size": HID_SIZE,
    "device": DEVICE,
    # "batch_size": BATCH_SIZE,
    "num_workers": NUM_WORKERS,
}

orig_magn_3_5_params = {
    **default_params,
    "batch_size": 32,
    "heavy_quake_thres": 3.5,
    "celled_data_x_path": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "celled_data_y_path": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "celled_data_path_for_freq_map": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "model_name": "model_maxmagn_Xorig_Yorig_magn3_5",
    "n_cycles": 5,
    "learning_rate": 2e-4,
    "earthquake_weight": 10e7,
    "lr_decay": 10.0,
    "start_lr_decay": 110.0,
    "weight_decay": 1e-6,
}

orig_magn_6_params = {
    **default_params,
    "heavy_quake_thres": 6.0,
    "celled_data_x_path": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "celled_data_y_path": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "celled_data_path_for_freq_map": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "model_name": "model_maxmagn_Xorig_Yorig_magn6",
    "n_cycles": 25,
    "learning_rate": 0.006,
    "non_earthquake_weight": 0.001,
    "earthquake_weight": 0.999,
    "lr_decay": 0.9,
    "start_lr_decay": 0,
    "weight_decay": 5.0,
    "retrain": False,
    "min_best_epoch": 1,
}

without_aft_magn_3_5_params = {
    **default_params,
    "batch_size": 32,
    "heavy_quake_thres": 3.5,
    "celled_data_x_path": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "celled_data_y_path": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "celled_data_path_for_freq_map": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN,
    "model_name": "model_withoutaft_magn3_5",
    "n_cycles": 16,
    "learning_rate": 0.01,
    "non_earthquake_weight": 0.001,
    "earthquake_weight": 0.999,
    "lr_decay": 0.5,
    "start_lr_decay": 1,
    "weight_decay": 0.5,
    "retrain": False,
}

without_aft_magn_6_params = {
    **default_params,
    "heavy_quake_thres": 6.0,
    "celled_data_x_path": CELLED_DATA_PATH_CAT_ORIG_MAXMAGN_DENSITY,
    "celled_data_y_path": CELLED_DATA_PATH_WITHOUTAFT_CAT_MAXMAGN,
    "celled_data_path_for_freq_map": CELLED_DATA_PATH_WITHOUTAFT_CAT_MAXMAGN,
    "model_name": "model_maxmagn_density_Xorig_Ywithoutaft_magn6",
    "n_cycles": 10,
    "learning_rate": 0.9,
    "earthquake_weight": 0.9976,
    "non_earthquake_weight": 0.0022,
    "lr_decay": 0.5,
    "start_lr_decay": 1,
    "weight_decay": 0.1,
    "retrain": False,
    "min_best_epoch": 1,
}
