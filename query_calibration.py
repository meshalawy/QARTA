#%%
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from pyproj.enums import TransformDirection
from tqdm.auto import tqdm


#%%
trips_unfiltered = pd.read_csv('data/nyc_1k_zoned.csv')

# calculate geodesic distance (straight line distance on earth surface)
trips_unfiltered['sl'] = trips_unfiltered.apply(
    lambda x : geodesic(
        x[["PickupLat","PickupLon"]].tolist(), 
        x[["DropLat","DropLon"]].tolist()).km
    , axis=1)

# duration from minutes to seconds 
trips_unfiltered.GT_Duration = trips_unfiltered.GT_Duration * 60
# %%

min_duration = 3 * 60
max_duration = 120 * 60

filter_out = (
    (trips_unfiltered.GT_Duration < min_duration) |
    (trips_unfiltered.GT_Duration > max_duration) |
    (trips_unfiltered.SourceZone == -1)           |
    (trips_unfiltered.DestZone == -1)             |
    (trips_unfiltered.OsrmDistance == 0)          |
    # (trips_unfiltered.QARTA_Distance == 0)        |
    (trips_unfiltered.sl < 1)         
)
trips = trips_unfiltered[~filter_out]


# %%
# Features extraction:
trips['TripStartTime'] = pd.to_datetime(trips.TripStartTime, format = '%m-%d-%y %H:%M')
trips['hour_of_day'] = trips.TripStartTime.dt.hour + 1
trips['day_of_week'] = trips.TripStartTime.dt.weekday + 1
trips['hour_of_week'] = trips.TripStartTime.dt.weekday * 24 +  \
                        trips.TripStartTime.dt.hour + 1

trips['sl_OSM_ratio'] = trips.apply(lambda x: x.sl/min(1,x.OsrmDistance), axis=1)
# trips['sl_QARTA_ratio'] = trips.apply(lambda x: x.sl/min(1,x.QARTA_Distance), axis=1)
trips['OSM_diff'] = trips.GT_Duration - trips.OsrmDuration
# trips['QARTA_diff'] = trips.GT_Duration - trips.QARTA_Duration






# %%
# Training & Testing & Evaluation Metrics

from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

# metrics
def ae_score(y_true, y_pred):
	return np.abs(y_true - y_pred) 

def pe_score(y_true, y_pred):
	return (np.abs((y_true - y_pred) / y_true)) * 100

def mape_score(y_true, y_pred):
	return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def medape_score(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

    
def train_test (config, X_train, Y_train, X_test, Y_test):
    mod = GradientBoostingRegressor(n_estimators=300, max_depth=5)
    mod.fit(X_train, Y_train)
    yhat = mod.predict(X_test)
    

    # when using diff
    if config['is_offset_based']:
        
        yhat += X_test[config['offset']['target_before']]
        Y_test += X_test[config['offset']['target_before']]
        
    
    print("",
        f'{config["name"]},Mean AE, {mean_absolute_error(Y_test, yhat)}', '\n',
        f'{config["name"]},Median AE, {median_absolute_error(Y_test, yhat)}','\n',
        f'{config["name"]},Mean PE%, {mape_score(np.array(Y_test), np.array(yhat))}','\n',
        f'{config["name"]},Median PE%, {medape_score(np.array(Y_test), np.array(yhat))}','\n'
    )

    if config['target2']:
        yhat2=trips.loc[test_idx][config['target2']]
        print("",
            f'{config["name2"]},Mean AE, {mean_absolute_error(Y_test, yhat2)}', '\n',
            f'{config["name2"]},Median AE, {median_absolute_error(Y_test, yhat2)}','\n',
            f'{config["name2"]},Mean PE%, {mape_score(np.array(Y_test), np.array(yhat2))}','\n',
            f'{config["name2"]},Median PE%, {medape_score(np.array(Y_test), np.array(yhat2))}','\n'
        )

        model_errors = pd.DataFrame({
            f'{config["name"]} predicted' : yhat,
            f'{config["name"]} PE' : pe_score(np.array(Y_test), np.array(yhat)),
            f'{config["name"]} AE' : ae_score(np.array(Y_test), np.array(yhat)),
            f'{config["name2"]} predicted' : yhat2,
            f'{config["name2"]} PE' : pe_score(np.array(Y_test), np.array(yhat2)),
            f'{config["name2"]} AE' : ae_score(np.array(Y_test), np.array(yhat2))
        }, index = X_test.index)
    
        return model_errors
    
    else:
        model_errors = pd.DataFrame({
            f'{config["name"]} predicted' : yhat,
            f'{config["name"]} PE' : pe_score(np.array(Y_test), np.array(yhat)),
            f'{config["name"]} AE' : ae_score(np.array(Y_test), np.array(yhat)),
            
        }, index = X_test.index)
    
        return model_errors


# %%
from sklearn.model_selection import train_test_split


config1 = {
    "features": ['hour_of_day','day_of_week','hour_of_week', 'OsrmDistance', 'OsrmDuration','SourceZone','DestZone','sl'],
    "categorical": [],

    "target": 'OSM_diff',
    "name": 'Q-Calib.',
    "is_offset_based": True,
    "offset": {
        'target_before': 'OsrmDuration',
        'target_after': 'GT_Duration'
    },
    
    "target2": 'OsrmDuration',
    "name2": 'OSM'
}

config2 = {
    "features": ['hour_of_day', 'day_of_week', 'hour_of_week', 'QARTA_Distance', 'QARTA_Duration', 'SourceZone','DestZone','sl'],
    "categorical": [],

    "target": 'QARTA_diff',
    "name": 'QARTA',

    "is_offset_based": True,
    "offset": {
        'target_before': 'QARTA_Duration',
        'target_after': 'GT_Duration'
    },


    "target2": 'QARTA_Duration',
    "name2": 'Q-Map'
}


# configs2run = [config1, config2]
configs2run = [config1]


# Splitting data
train, test = train_test_split(trips,test_size=0.25)
train_idx = train.index
test_idx = test.index
results = pd.DataFrame(index=test_idx)
time_results = []

# run and measure time
import time

for config in configs2run:
    X = pd.get_dummies(trips[config['features']], columns=config['categorical'])
    Y = trips[config['target']]


    X_train = X.loc[train_idx]
    Y_train = Y.loc[train_idx]
    X_test = X.loc[test_idx]
    Y_test = Y.loc[test_idx]
    
    start_time = time.time()
    results = pd.concat([results,
                        train_test(config, X_train, Y_train, X_test, Y_test)
                        ], axis=1)

    time_results.append({
        "name":config['name'],
        "time": time.time() - start_time
    })


print(time_results)

results.agg({ 
    x:['mean','median'] 
        for x in results.columns 
            if x.endswith('PE')}).T.plot(kind='bar')

results.agg({ 
    x:['mean','median'] 
        for x in results.columns 
            if x.endswith('AE')}).T.plot(kind='bar')


# %%



# accuracy distribution over distance 
test_data = trips.loc[test_idx]

# results['QARTA_Distance'] = test_data.QARTA_Distance
results['OsrmDistance'] = test_data.OsrmDistance
test_data = test_data[test_data.OsrmDistance < 50000]
results.groupby([test_data.OsrmDistance // 5000]).agg({
    "OSM PE" : 'median',
    'Q-Calib. PE': 'median'
}).plot()

# %%
