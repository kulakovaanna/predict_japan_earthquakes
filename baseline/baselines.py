
import pandas as pd
import numpy as np
import datetime
from dateutil.relativedelta import relativedelta
from tqdm import tqdm

from utils import create_celled_data
from constants import N_CELLS_HOR, N_CELLS_VER, BBOX


class baseline1():
    def __init__(
        self, df, horizon_min=10, horizon_max=50, testing_days=1000,
        n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, bbox=BBOX
    ):
        self.df = df
        self.horizon_min_dt = datetime.timedelta(days=horizon_min)
        self.horizon_max_dt = datetime.timedelta(days=horizon_max)
        self.test_end_dt = self.df["time"].max()
        self.test_start_dt = self.test_end_dt - datetime.timedelta(days=testing_days)
        self.train_start_dt = self.test_start_dt - relativedelta(years=10)
        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver
        self.bbox = bbox
        
        print(f"{self.__class__.__name__}\nhorizon_min: {horizon_min}, horizon_max: {horizon_max}, testing_days: {testing_days}")
        
    def predict(self):
        train_df = self.df[
            (self.df["time"] >= self.train_start_dt) &
            (self.df["time"] < self.test_start_dt)
        ]
        test_df = self.df[self.df["time"] >= self.test_start_dt]
        
        baseline_density_map, _, _ = create_celled_data(
            lon=train_df["lon"], lat=train_df["lat"], 
            n_cells_hor=self.n_cells_hor, n_cells_ver=self.n_cells_ver, bbox=self.bbox
        )

        target = []
        current_dt = self.test_start_dt
        total_days = ((self.test_end_dt - self.horizon_max_dt) - self.test_start_dt).days
        with tqdm(total=total_days) as pbar:
            while current_dt < (self.test_end_dt - self.horizon_max_dt):
                sample_df = test_df[
                        (test_df["time"] >= (current_dt + self.horizon_min_dt)) &
                        (test_df["time"] <= (current_dt + self.horizon_max_dt))
                    ]
                target_map, _, _ = create_celled_data(
                    lon=sample_df["lon"], lat=sample_df["lat"], n_cells_hor=N_CELLS_HOR, 
                    n_cells_ver=N_CELLS_VER, bbox=BBOX, density=False
                )
                target.append(target_map)
                
                current_dt += datetime.timedelta(days=1)
                pbar.update(1)

        target = np.asarray(target)
        prediction = np.asarray([baseline_density_map]*len(target))
        
        return target, prediction
    
class baseline2():
    def __init__(
            self, df, horizon_min=10, horizon_max=50, testing_days=1000, window_years=5, 
            n_cells_hor=N_CELLS_HOR, n_cells_ver=N_CELLS_VER, bbox=BBOX
        ):
        self.df = df
        self.test_end_dt = self.df["time"].max()
        self.test_start_dt = self.test_end_dt - datetime.timedelta(days=testing_days)
        self.horizon_min_dt = datetime.timedelta(days=horizon_min)
        self.horizon_max_dt = datetime.timedelta(days=horizon_max)
        self.window = window_years
        self.n_cells_hor = n_cells_hor
        self.n_cells_ver = n_cells_ver
        self.bbox = bbox
        
        print(f"{self.__class__.__name__}\nhorizon_min: {horizon_min}, horizon_max: {horizon_max}, window_years: {window_years}, testing_days: {testing_days}")
        
    def predict(self):
        target = []
        prediction = []
        current_dt = self.test_start_dt

        total_days = ((self.test_end_dt - self.horizon_max_dt) - self.test_start_dt).days
        with tqdm(total=total_days) as pbar:
            while current_dt < (self.test_end_dt - self.horizon_max_dt):
                train_df = self.df[
                    (self.df["time"] >= (current_dt - relativedelta(years=self.window))) &
                    (self.df["time"] <= current_dt)
                ]
                test_df = self.df[
                    (self.df["time"] >= (current_dt + self.horizon_min_dt)) &
                    (self.df["time"] <= (current_dt + self.horizon_max_dt))
                ]
                
                baseline_density_map, _, _ = create_celled_data(
                    lon=train_df["lon"], lat=train_df["lat"], 
                    n_cells_hor=self.n_cells_hor, n_cells_ver=self.n_cells_ver, bbox=self.bbox
                )
                target_map, _, _ = create_celled_data(
                    lon=test_df["lon"], lat=test_df["lat"], density=False,
                    n_cells_hor=self.n_cells_hor, n_cells_ver=self.n_cells_ver, bbox=self.bbox
                )
                target.append(target_map)
                prediction.append(baseline_density_map)
                
                current_dt += datetime.timedelta(days=1)
                pbar.update(1)
            
        target = np.asarray(target)
        prediction = np.asarray(prediction)
        
        return target, prediction