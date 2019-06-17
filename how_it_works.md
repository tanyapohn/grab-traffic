# Feature Engineering

### Based Features (1st model)

+ *day_mod* - take a modulo from day and gives the day range in [1,31]
+ *week_day* - day of week, to see the pattern of deamnd in a week
+ *consec_zeros* - to capture how long does demand take to change from 0s
+ *hour* - hour extracted from timestamp
+ *minute* - minute extracted from timestamp
+ *mean_demand_per_hour* - an average of demand of each hour in a week
+ *median_demand_per_hour* - a median of demand of each hour in a week
+ *mean_demand_per_week* - an average of demand in a week
+ *median_demand_per_week* - a median of demand in a week
+ *mean_demand_per_geo* - an average of demand of each geohash6 in a week
+ *median_demand_per_geo* - a median of demand of each geohash6 in a week
+ *x_coord* - an x coordinate extracted from geohash6
+ *y_coord* - a y coordinate extracted from geohash6
+ *z_coord* - an z coordinate extracted from geohash6

To convert geohash6 in x, y, z coordinates is more meaningful than using geohash6 alone in regression 

### Diff Features (2nd model)

After the demand had been predicted *(y_pred)* by 1st model, this y_pred
will be *shifted back for 1 step*, *moved average for 3 windows* and taken a *difference* between 
those `mean_*` and `median_*` features. It generate the following features:

**Shift 1:**

+ *u_diff_lag* - a difference of *mean_demand_per_hour* and *shifted* y_pred
+ *median_diff_lag* - a difference of *median_demand_per_hour* and *shifted* y_pred
+ *u_diff_lag_week* - a difference of *mean_demand_per_week* and *shifted* y_pred
+ *median_diff_lag_week* - a difference of *median_demand_per_week* and *shifted* y_pred
+ *u_diff_lag_geo* - a difference of *mean_demand_per_geo* and *shifted* y_pred
+ *median_diff_lag_geo* - a difference of *median_demand_per_geo* and *shifted* y_pred

**Moving average:**

+ *u_diff_rolling* - a difference of *mean_demand_per_hour* and *moving average* y_pred
+ *u_diff_rolling_week* - a difference of *mean_demand_per_week* and *moving average* y_pred
+ *u_diff_rolling_geo* - a difference of *mean_demand_per_geo* and *moving average* y_pred

The reason to take a difference among them is to capture the pattern of a week and hour.

# Model

I create 2 models for this project

- First, it is an Xgboost model producing a base demand from *geohash6, day and timestamp*.
I would call it *y_pred*

After *y_pred* has been create, this field will be sent to the second model.

- My second model will take *y_pred* as a feature tranformed to those `Diff Features` 
that I mentioned earlier. This model will predict **T+1** at the first step 
and **T+5** afterwards.

- To predict **T+5**, the demand value must have been predicted at T+2, T+3 and T+4 before
as this following equation:

```bash
    T     =    1st_model(based_features)
    T+1   =    2nd_model(T)
    T+2   =    2nd_model(T+1)
    .
    .
    .
    T+5   =    2nd_model(T+4)
``` 

The reason I choose Xgboost is because the limitation of my RAM and time resource. 
Since the data is too big, without fine tuning, Xgboost is the good choice of pruning 
and boosting the regression tree.