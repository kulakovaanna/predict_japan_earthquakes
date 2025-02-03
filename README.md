# predict_japan_earthquakes

In the predict_japan_earthquakes repository represented code for naive baselines, classical ml and lstm models for predicting earthquakes in Japan, utilising two earthquake catalogues containing data from 2004 to 2023. One catalogue includes data with aftershocks, while the other excludes them. The data has been thresholded at magnitudes 3.5 and 6, as the prediction of earthquakes of higher magnitude is a more challenging undertaking.

The lstm model is based on the lstm model from the paper "Recurrent Convolutional Neural Networks help to predict location of Earthquakes" (Roman Kail, Alexey Zaytsev, Evgeny Burnaev).

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
    │   ├── utils.py                                    -> utilities for classical ml models
    │   ├── ml_utils.py                                 /
    │   ├── merge_features.ipynb                        -> notebook for merging collected features via collect_dataset.py in datasets
    │   ├── ml-models_magn_3_5.ipynb                    -> boosting models for earthquakes harder than magnitude 3.5
    │   └── validation.ipynb                            -> validating result boosting models
    └── lstm_models
        ├── Dataset_creation_orig_paper.ipynb           -> dataset creation for lstm model from original paper
        ├── Dataset_creation_two_catalogs.ipynb         -> dataset creation for two datasets (data/catalogues/*.csv)
        ├── experiments                                 -> notebooks with experiments
        │   ├── Learning_orig_paper.ipynb
        │   ├── Learning_two_catalogs_magn3_5.ipynb
        │   └── Learning_two_catalogs_magn6.ipynb
        ├── config.py                                   -> config file with constants and hyperparameteres
        ├── model.py                                    -> implementation of models
        ├── result_models.ipynb                         -> notebook with result lstm models
        ├── train.py                                    -> train utilities
        └── utils.py                                    -> utilities

## datasets

- source dataset with aftershocks: `https://disk.yandex.ru/d/KpWoteZyrx8nyA`
- source dataset without aftershocks: `https://disk.yandex.ru/d/UKgbIfYor1R84Q`
- dataset for classic ml models: `https://disk.yandex.ru/d/cSx_6Ek01x_LhQ`
