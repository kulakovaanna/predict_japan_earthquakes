# predict_japan_earthquakes

In the predict_japan_earthquakes repository contains code for naive baselines, classical ml and lstm models for predicting earthquakes in Japan, utilising two earthquake catalogues containing data from 2004 to 2023. One catalogue includes data with aftershocks, while the other excludes them. The data has been thresholded at magnitudes 3.5 and 6, as the prediction of earthquakes of higher magnitude is a more challenging undertaking.

The lstm model is based on the lstm model from the paper "Recurrent Convolutional Neural Networks help to predict location of Earthquakes" (Roman Kail, Alexey Zaytsev, Evgeny Burnaev). We enhanced this approach by stacking the 5-year historical earthquake frequency with LSTM-derived adjustments, integrating physical seismicity patterns with data-driven temporal dynamics. To better capture the spatial variability and seismic energy release, we replaced binary input maps with earthquake density and maximum magnitude maps, which were selected based on empirical performance improvements.

## project structure

    ├── README.md
    ├── data
    │   ├── catalogues                                  -> source datasets (could be downloaded from cloud)
    │   │   ├── originalCat.csv                         -> dataset with aftershocks
    │   │   └── withoutAftCat.csv                       -> dataset without aftershocks
    │   └── jp_map.png
    ├── baseline                                        -> naive baselines
    │   ├── baseline.ipynb                              -> notebook with baselines
    │   ├── baselines.py                                -> implementation of baseline classes
    │   ├── constants.py
    │   └── utils.py
    ├── classic_models
    ├── data/models                                     -> trained models in pickle
        │   ├── *.pickle
    ├── data/dataset
        │   ├── *.parquet                               -> dataset for classic ml models in parquet (could be downloaded from cloud)
    │   ├── requirements.txt
    │   ├── constants.py
    │   ├── collect_dataset.py                          -> dataset creation for classical ml models
    │   ├── train.py                                    \
    │   ├── utils.py                                     -> utilities for classical ml models
    │   ├── ml_utils.py                                 /
    │   ├── merge_features.ipynb                        -> notebook for merging collected features via collect_dataset.py in datasets
    │   ├── logreg_magn_3_5.ipynb                       -> logreg models for earthquakes harder than magnitude 3.5
    │   ├── logreg_magn_6.ipynb                         -> logreg models for earthquakes harder than magnitude 6.0
    │   ├── logreg_validation.ipynb                     -> validating result logreg models
    │   ├── boosting_magn_3_5.ipynb                     -> boosting models for earthquakes harder than magnitude 3.5
    │   ├── boosting_magn_6.ipynb                       -> boosting models for earthquakes harder than magnitude 6.0
    │   └── boosting_validation.ipynb                   -> validating result boosting models
    └── lstm_models
        ├── models                                      -> trained models
        │   ├── *.pth
        ├── Dataset_creation.ipynb                      -> dataset creation for lstm model
        ├── lstm_experiments.ipynb                      -> notebook with experiments with lstm models
        └── lstm_utils.py                               -> EarthquakeDataset and EarthquakePredictionModel

## datasets

- source dataset with aftershocks: `https://disk.yandex.ru/d/KpWoteZyrx8nyA`
- source dataset without aftershocks: `https://disk.yandex.ru/d/UKgbIfYor1R84Q`
- dataset for classic ml models: `https://disk.yandex.ru/d/cSx_6Ek01x_LhQ`
- lstm models weights: `https://disk.yandex.ru/d/vAyCdRG6yX8fcg`

