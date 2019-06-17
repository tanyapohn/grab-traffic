# Traffic Management Challenge

## Prerequisites

This project contains end-to-end processes of a model prediction.

To see my idea of creating features and models I described it [here](how_it_works.md)

My jupyter notebooks for EDA, creating features and modeling is in `notebooks` dir of this repo

## Environment Setup
1. This project uses the [pipenv](https://github.com/pyenv/pyenv-installer) as a virtual environment

2. Go to the root project and run the following:

    ```bash
    $ pipenv install
    ``` 
3. To start a virtual environment, run the following:

    ```bash
    $ pipenv shell
    ```

## Main Files

For  training steps:
- `gen_training_feats.py`: Create based features from training set which produces
features such as mean and median based on time interval, coverts geohash6 to x, y, z coordinates
and a consecutive values of 0s demand

- `gen_training_feats_2.py`: After the based features had been created, this method allows to generate
the difference between moving average and based features.

- `trainer.py`: Train the model

To run the predictor:

``` bash
$ python predictor.py --path [test_set_path] 
```

where *[test_set_path]* is a specified absolute path of your _test_set.csv_.
This predictor produces the results of both **T+1** and **T+5** stored under `results` directory of this project.

- `predictor.py` also automatically generates all features

## File Structure
```bash
.
+-- data
|    +-- diff_features
|    |   +-- diff_feats.csv
|    +-- gen_features
|    |   +-- checkpoint_1
|    |   +-- checkpoint_2
|    +-- u_median_features
|    |   +-- consec_zeros.csv
|    |   +-- median_demand_geo.csv
|    |   +-- median_demand_time.csv
|    |   +-- median_demand_week.csv
|    |   +-- u_demand_geo.csv
|    |   +-- u_demand_time.csv
|    |   +-- u_demand_week.csv
|    +-- training.csv
+-- funcs
|    +-- geo_features.py
|    +-- helper.py
|    +-- prepare_train.py
|    +-- split_data.py
|    +-- time_features.py    
+-- model
|    +-- xgb_t_plus_1.model
|    +-- xgb_t_plus_1_2.model
+-- results
+-- gen_feats.py
+-- gen_training_feats.py
+-- gen_training_feats_2.py
+-- predictor.py
+-- trainer.py
```
