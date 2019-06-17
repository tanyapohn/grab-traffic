# Traffic Management Challenge

## Prerequisites

This project contains end-to-end processes of a model prediction.

To see my idea of creating features and models I described it [here](how_it_works.md)

My jupyter notebooks for EDA, creating features and modeling are in [notebooks](notebooks) dir  of this repo.
Since the code in notebooks is not very clean, I would recommend you to look at it in `*.py` instead

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

To run the predictor, after initiate the virtual environment:

``` bash
$ python predictor.py --path [test_set_path] 
```

where *[test_set_path]* is a specified absolute path of your _test_set.csv_ in string format.
This predictor produces the results of both **T+1** and **T+5** stored under `results` directory of this project.

- `predictor.py` also automatically generates all features

For  training steps:
- `gen_training_feats.py`: Create based features for 1st model from training set which produces
features such as mean and median based on time interval, coverts geohash6 to x, y, z coordinates
and a consecutive values of 0s demand

- `gen_training_feats_2.py`: After the based features had been created, this method allows to generate
the difference between moving average and based features for 1nd model.

- `trainer.py`: Train the models

- `training_T_plus_1_xgb.ipynb`, `training_T_plus_5_xgb.ipynb` : Train the 1st model and 2nd model in jupyter notebook 
*(It's not very clean, but I want to show you how I trained them)*


## File Structure
```bash
.
+-- data
|   +-- diff_features
|   |   +-- diff_feats.csv
|   +-- gen_features
|   |   +-- checkpoint_1
|   |   +-- checkpoint_2
|   +-- u_median_features
|   |   +-- consec_zeros.csv
|   |   +-- median_demand_geo.csv
|   |   +-- median_demand_time.csv
|   |   +-- median_demand_week.csv
|   |   +-- u_demand_geo.csv
|   |   +-- u_demand_time.csv
|   |   +-- u_demand_week.csv
|   +-- training.csv
+-- funcs
|   +-- geo_features.py
|   +-- helper.py
|   +-- prepare_train.py
|   +-- split_data.py
|   +-- time_features.py    
+-- model
|   +-- xgb_t_plus_1.model
|   +-- xgb_t_plus_1_2.model
+-- notebooks
|   +-- make_features_tree.ipynb
|   +-- modeling_make_feats_T_plus_1.ipynb
|   +-- traffic_model_playground.ipynb
|   +-- training_T_plus_1_xgb.ipynb
|   +-- training_T_plus_5_xgb.ipynb
+-- results
+-- gen_feats.py
+-- gen_training_feats.py
+-- gen_training_feats_2.py
+-- predictor.py
+-- trainer.py
```
