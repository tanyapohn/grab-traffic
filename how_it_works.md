# Feature Engineering

### Based Features (1st model):

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

### Diff Features (2nd model)

After the demand had been predicted *(y_pred)* by 1st model, this y_pred
will be *shifted back for 1 step*, *moved average for 3 windows* and taken a *difference* between 
those `mean_*` and `median_*` features. It generate the following features:

Shift 1:

+ *u_diff_lag* - a difference of *mean_demand_per_hour* and **shifted** y_pred
+ *median_diff_lag* - a difference of *median_demand_per_hour* and **shifted** y_pred
+ *u_diff_lag_week* - a difference of *mean_demand_per_week* and **shifted** y_pred
+ *median_diff_lag_week* - a difference of *median_demand_per_week* and **shifted** y_pred
+ *u_diff_lag_geo* - a difference of *mean_demand_per_geo* and **shifted** y_pred
+ *median_diff_lag_geo* - a difference of *median_demand_per_geo* and **shifted** y_pred

Moving average:

+ *u_diff_rolling* - a difference of *mean_demand_per_hour* and **moving average** y_pred
+ *u_diff_rolling_week* - a difference of *mean_demand_per_week* and **moving average** y_pred
+ *u_diff_rolling_geo* - a difference of *mean_demand_per_geo* and **moving average** y_pred

The reason to take a difference among them is to capture the seasonality of a week and hour.

# Model
