# lstm_earthquakes

The model is based on the lstm model from the paper "Recurrent Convolutional Neural Networks help to predict location of Earthquakes" (Roman Kail, Alexey Zaytsev, Evgeny Burnaev).

## project structure

    ├── README.md
    ├── baseline                                        -> naive baselines
    │   ├── baseline.ipynb                              -> notebook with baselines
    │   ├── baselines.py                                -> implementation of baseline classes
    │   ├── constants.py
    │   └── utils.py
    ├── data
    │   ├── catalogues
    │   │   ├── originalCat.csv                         -> dataset without aftershocks
    │   │   └── withoutAftCat.csv                       -> dataset with aftershocks
    │   └── jp_map.png
    └── src
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

- dataset with aftershocks: `https://disk.yandex.ru/d/KpWoteZyrx8nyA`
- dataset without aftershocks: `https://disk.yandex.ru/d/UKgbIfYor1R84Q`