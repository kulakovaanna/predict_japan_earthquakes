import warnings
import numbers
import re
from copy import deepcopy
from pathlib import Path
import pickle
from typing import Any
import typing as tp
from contextlib import redirect_stdout
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer, make_column_selector
import lightgbm as lgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from probatus.feature_elimination import ShapRFECV
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
import optuna
import optuna.logging
import optuna.study
import optuna.trial

from ml_utils import MonthlyTimeSeriesSplit, get_dt_from_index, weighted_roc_metric, weighted_roc_metric_harder_magn

optuna.logging.set_verbosity(optuna.logging.WARNING)


def get_logreg_model(features: list[str], params: dict[str, float | int] | None = None) -> Pipeline:
    if params is None:
        params = {
            "max_iter": 1_000,
        }
    
    return make_pipeline(
        make_column_transformer(
            ("passthrough", features),
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        make_column_transformer(
            (
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                make_column_selector(dtype_include="number"),
            ),
            remainder="passthrough",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        make_column_transformer(
            (
                StandardScaler(),
                make_column_selector(dtype_include="number"),
            ),
            remainder="passthrough",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        LogisticRegression(**params),
    )


def get_lgbm_model(features: list[str], params: dict[str, float | int] | None = None) -> Pipeline:
    if params is None:
        params = {
            "n_estimators": 50,
            "objective": "binary",
            "use_missing": False,
            "deterministic": True,
            "random_state": 42,
            "force_col_wise": True,
            "feature_pre_filter": False,
            "verbosity": -1,
            "max_depth": 4,
            "n_jobs": 40,
        }
    return make_pipeline(
        make_column_transformer(
            ("passthrough", features),
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        make_column_transformer(
            (
                SimpleImputer(strategy="constant", fill_value=0.0, keep_empty_features=True),
                make_column_selector(dtype_include="number"),
            ),
            remainder="passthrough",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        ).set_output(transform="pandas"),
        lgb.LGBMClassifier(**params),
    )
    
    
def fixed_params(**kwargs) -> dict[str, Any]:
    params = {
        "objective": "binary",
        "use_missing": False,
        "deterministic": True,
        "random_state": 42,
        "force_col_wise": True,
        "feature_pre_filter": False,
        "verbosity": -1,
        "n_jobs": 1,
    }

    params.update(kwargs)

    return params


def default_params(**kwargs) -> dict[str, Any]:
    params = {
        **fixed_params(),
        "colsample_bytree": 1.0,
        "subsample": 1.0,
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": 31,
        "min_child_samples": 20,
        "n_extimators": 30,
    }
    params.update(kwargs)

    return params


def suggest_params(trial: optuna.trial.Trial, **kwargs) -> dict[str, Any]:
    params = {
        **fixed_params(),
        "n_estimators": trial.suggest_int("n_estimators", 10, 200, step=5),
        "max_depth": trial.suggest_int("max_depth", 1, 10),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "subsample": trial.suggest_float("subsample", 0.1, 1.0),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, step=0.001),
        "num_leaves": trial.suggest_int("num_leaves", 10, 200),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 50),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.8, 5.0),
    }
    params.update(kwargs)

    return params

def run_optuna(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    cv,
    n_trials: int = 1000,
    dump_study_path: tp.Optional[str] = None,
    seed: int = 42,
    direction: str = "maximize",
    resume: bool = False,
    show_progress: bool = True,
    suggest_params: dict = suggest_params,
    default_params: dict = default_params,
    model=None,
    n_jobs: int = 10,
    model_name: str = "model",
    harder_magn: bool = False,
):
    if isinstance(cv, MonthlyTimeSeriesSplit):
        print("using MonthlyTimeSeriesSplit")
        
    model_ = deepcopy(model)

    def objective(trial: optuna.trial.Trial):
        params = suggest_params(trial)
        model = deepcopy(model_.set_params(**{f"{model_name}__" + k: v for k, v in params.items()}))

        pred = cross_val_predict(
            model,
            X,
            y,
            cv=cv,
            n_jobs=n_jobs,
            method="predict_proba",
        )
        pred = pd.Series(pred[:, 1], index=y.index)
        
        if isinstance(cv, MonthlyTimeSeriesSplit):
            dt = get_dt_from_index(X)
            min_test_date = dt[next(cv.split(X))[1]].min()
            is_test = dt >= min_test_date
            # score = roc_auc_score(y[is_test], pred[is_test])
            if harder_magn:
                score = weighted_roc_metric_harder_magn(y[is_test], pred[is_test])
            else:
                score = weighted_roc_metric(y[is_test], pred[is_test])
        else: 
            # score = roc_auc_score(y, pred)
            if harder_magn:
                score = weighted_roc_metric_harder_magn(y, pred)
            else:
                score = weighted_roc_metric(y, pred)

        return float(score)

    callbacks = []
    if dump_study_path is not None:

        def dump_study(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            with open(dump_study_path, "wb") as f:
                pickle.dump(study, f, protocol=4)

        callbacks.append(dump_study)

    if resume and dump_study_path is not None and Path(dump_study_path).exists():
        with open(dump_study_path, "rb") as f:
            study = pickle.load(f)
        n_trials = max(0, n_trials - len(study.get_trials(deepcopy=False)))
    else:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction=direction, sampler=sampler)
        study.enqueue_trial(default_params())

    if show_progress:
        pbar = tqdm(position=0, desc=f"{model_name} hyperoptimisation", total=n_trials)

        def update_pbar(study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
            pbar.update()

        callbacks.append(update_pbar)

    study.optimize(
        objective,
        n_trials=n_trials,
        callbacks=callbacks,
    )

    best_params = suggest_params(study.best_trial)
    best_model = model_.set_params(**{f"{model_name}__" + k: v for k, v in best_params.items()})

    return best_model
    

def get_shaprfecv_n_steps(*, n_features: int, step) -> int:
    if isinstance(step, numbers.Integral):
        return 1 + (n_features + step - 2) // step
    l = n_features
    n_steps = 0
    while l > 0:
        n_removed = max(1, int(l * step))
        l -= n_removed
        n_steps += 1

    return n_steps


class ShapRFECVProgressTracker:
    def __init__(self, pbar):
        self.pbar = pbar

    def write(self, message):
        if message.startswith("Round:"):
            match = re.search("(?<=Round:)\s*[0-9]+(?=, Current number of features:)", message)
            if match is not None:
                step = int(match[0])
                self.pbar.update(max(0, step - self.pbar.n))
                self.pbar.refresh()

    def flush(self):
        pass


def select_features(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv = 10,
    n_jobs: int =10,
    metric="roc_auc",
    step: float = 0.1,
    min_features_to_select: int = 10,
    columns_to_keep: list[str] | None = None,
    *,
    show_progress: bool = True,
    return_rfe: bool = False,
    verbose : int = 60,
):
    shap_elimination = ShapRFECV(
        model=model,
        step=step,
        cv=cv,
        min_features_to_select=min_features_to_select,
        scoring=metric,
        n_jobs=n_jobs,
        random_state=42,
        verbose=verbose,
    )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        with tqdm(
            position=0,
            total=get_shaprfecv_n_steps(n_features=X.shape[1], step=step),
            desc="feature selection",
            disable=(not show_progress),
        ) as pbar:
            with redirect_stdout(ShapRFECVProgressTracker(pbar)):
                rfe_df = shap_elimination.fit_compute(
                    X,
                    y,
                    columns_to_keep=columns_to_keep,
                    groups=None,
                    check_additivity=False,  # needed due to a bug in computing shap values
                )
    best_row = rfe_df.loc[rfe_df[rfe_df["val_metric_mean"] == rfe_df["val_metric_mean"].max()]["num_features"].idxmin()]
    best_num_features = int(best_row["num_features"])
    best_features = list(shap_elimination.get_reduced_features_set(num_features=best_num_features))

    if return_rfe:
        return rfe_df, best_features
    else:
        return best_features
