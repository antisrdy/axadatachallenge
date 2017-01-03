# coding=utf-8

import numpy as np
import sys
import csv
import pandas as pd
from importlib import import_module

n_burn_in = 672
# y_train array is blocked in the following way:
# n_burn_in | n_common_block | n_cv x block_size
# first cv fold takes 0 blocks (so only n_burn_in and n_common_block)
# last cv fold takes n_burn_in + n_common_block + (n_cv - 1) x block_size
# block_size should be divisible by 12 if score varies within year ->
# n_validation = n_train - int(n_train / 2) = n_train / 2
# should be divisible by 12 * n_cv -> n_train should be
# multiple of 24 * n_cv
# n_train should also be > 2 * n_burn_in
n_cv = 4

date_train = np.datetime64('2012-08-01 00:00:00.000')
date_train_y = np.datetime64('2012-08-08 00:00:00.000')
end_train = np.datetime64('2012-12-21 00:00:00.000')
end_train_y = np.datetime64('2012-12-28 00:00:00.000')

list_comp = [(np.datetime64('2013-01-03T23:30:00.000'), np.datetime64('2013-02-02T00:30:00.000')), 
             (np.datetime64('2013-02-08T23:30:00.000'), np.datetime64('2013-03-06T00:30:00.000')), 
             (np.datetime64('2013-03-12T23:30:00.000'), np.datetime64('2013-04-10T00:30:00.000')), 
             (np.datetime64('2013-04-16T23:30:00.000'), np.datetime64('2013-05-13T01:30:00.000')), 
             (np.datetime64('2013-05-19T23:30:00.000'), np.datetime64('2013-06-12T00:30:00.000')), 
             (np.datetime64('2013-06-18T23:30:00.000'), np.datetime64('2013-07-16T00:30:00.000')), 
             (np.datetime64('2013-07-22T23:30:00.000'), np.datetime64('2013-08-15T00:30:00.000')), 
             (np.datetime64('2013-08-21T23:30:00.000'), np.datetime64('2013-09-14T00:30:00.000')), 
             (np.datetime64('2013-09-20T23:30:00.000'), np.datetime64('2013-10-18T00:30:00.000')), 
             (np.datetime64('2013-10-24T23:30:00.000'), np.datetime64('2013-11-20T00:30:00.000')), 
             (np.datetime64('2013-11-26T23:30:00.000'), np.datetime64('2013-12-22T00:30:00.000'))]


def check_dataframe(X_ds):
    # Some constants
    timedelta = np.datetime64('2013-01-28 00:30:00.000') - np.datetime64('2013-01-28 00:00:00.000')
    startdate = np.datetime64('2011-01-01 00:00:00.000')
    enddate = np.datetime64('2013-12-31 23:30:00.000')

    #Dates that are present in the dataframe
    gotten_values = X_ds.index.values
    gotten_values.sort()

    #Dates that are not in the dataframe
    added_values = []

    #counter will parcour all the dates between startdate and enddate
    counter = startdate

    #indice is where we are in the sortted array of gotten values
    indice = 0

    #Loop over all dates, if they are present, increase counter and indice, 
    #else, put them in added_values and increase only counter
    while counter <= enddate:
        if counter == gotten_values[indice]:
            counter += timedelta
            indice += 1
        else:
            if counter > gotten_values[indice]:
                print("ERROR")
                break
            else:
                added_values.append(counter)
                counter += timedelta

    #  creation of a dataframe with all new values
    df_temp = pd.DataFrame.from_dict({"DATE" : added_values, "CSPL_RECEIVED_CALLS": np.zeros(len(added_values))})
    df_temp.index=df_temp["DATE"]
    df_temp = df_temp["CSPL_RECEIVED_CALLS"]

    #Concatenation with the old one

    result = pd.concat([X_ds, df_temp])
    result = result.sort_index()

    # Replace added_values by the mean of the previous and the following value
    # or by the previous value if the following one is in added_values
    for i, x in enumerate(added_values):
        if not i%1000:
            print( "value number " + str(i) )
        if x == startdate:
            result[x] = result[x+timedelta]
        elif (x+timedelta in added_values) | (x == enddate):
            result[x] = result[x-timedelta]
        else:
            result[x] = (result[x-timedelta] + result[x + timedelta])/2
    
    return result


def get_cv(y_train_array):
    n = len(y_train_array)
    n_common_block = int(n / 2)
    n_validation = n - n_common_block
    block_size = int(n_validation / n_cv)
    print('length of common block: %s half_hours = %s weeks' %
          (n_common_block, n_common_block / 336))
    print('length of validation block: %s half_hours = %s weeks' %
          (n_validation, n_validation / 336))
    print('length of each cv block: %s half_hours = %s weeks' %
          (block_size, block_size / 336))
    for i in range(n_cv):
        train_is = np.arange(n_common_block + i * block_size)
        test_is = np.arange(n_common_block + i * block_size, n_common_block + n_cv * block_size)
        yield (train_is, test_is)


def score(y_true, y_pred):
    return np.mean(np.exp(0.1 * (y_true - y_pred)) - 0.1 * (y_true  - y_pred) - 1)


def read_data(assignment):
    X_ds = []
    y_array = []
    
    file_name = assignment + ".csv"
    
    data = pd.read_csv(file_name, sep=";")
    X_ds = data["CSPL_RECEIVED_CALLS"]
    X_ds.index = data["DATE"].values.astype(np.datetime64)
    X_ds = check_dataframe(X_ds)
    y_array = X_ds.copy()
        
    return X_ds, y_array


def get_train_test_data(assignment):
    X_ds, y_array = read_data(assignment)

    X_train_ds = X_ds[X_ds.index<date_train]
    y_train_array = y_array[X_ds.index<date_train_y].iloc[1008::]
    print('length of training array: %s half hours = %s weeks' %
          (len(y_train_array), len(y_train_array) / 336))
    
    X_test_ds = X_ds[X_ds.index>date_train]
    X_test_ds = X_test_ds[X_test_ds.index<end_train]
    y_test_array = y_array[X_ds.index>date_train]
    y_test_array = y_test_array[y_test_array.index<end_train_y].iloc[1008::]
    print('length of test array: %s half hours = %s weeks' %
          (len(y_test_array), len(y_test_array) / 336))
    return X_train_ds, y_train_array, X_test_ds, y_test_array


def get_compl_data(assignment, list_ranges):
    X_ds, y_array = read_data(assignment) 

    timedelta = np.datetime64('2013-01-28 00:00:00.000') - np.datetime64('2013-01-21 00:00:00.000')

    ts_feature_extractor = import_module('ts_feature_extractor', '.')
    ts_fe = ts_feature_extractor.FeatureExtractor()

    X_comp = X_ds[X_ds.index>list_ranges[0][0]]
    X_comp = X_comp[X_comp.index<(list_ranges[0][1]-timedelta)]
    X_comp = ts_fe.transform(X_comp)

    y_comp = X_ds[X_ds.index>(list_ranges[0][0] + timedelta)]
    y_comp = y_comp[y_comp.index<list_ranges[0][1]]
    y_comp = y_comp.values[n_burn_in::]

    for a,b in list_ranges[1::]:
        X_temp = X_ds[X_ds.index>a] 
        X_temp = X_temp[X_temp.index<(b-timedelta)]

        X_comp = np.vstack((X_comp, ts_fe.transform(X_temp)))

        y_temp = X_ds[X_ds.index>(a + timedelta)]
        y_temp = y_temp[y_temp.index<b]

        y_comp = np.concatenate((y_comp, y_temp.values[n_burn_in::]))

    return X_comp, y_comp


def train_submission(module_path, X_ds, y_array, train_is, X_comp, y_comp):
    X_train_ds = X_ds[train_is]
    y_train_array = y_array[train_is].values[n_burn_in::]

    # Feature extraction
    ts_feature_extractor = import_module('ts_feature_extractor', module_path)
    ts_fe = ts_feature_extractor.FeatureExtractor()
    X_train_array = ts_fe.transform(X_train_ds)
    
    #Ajout du complément
    X_train = np.vstack((X_train_array, X_comp))
    y_train = np.concatenate((y_train_array, y_comp))

    # Regression
    regressor = import_module('regressor', module_path)
    reg = regressor.Regressor()
    reg.fit(X_train, y_train)
    return ts_fe, reg


def test_submission(trained_model, X_ds, test_is, X_comp=None):
    X_test_ds = X_ds[test_is]
    ts_fe, reg = trained_model
    # Feature extraction
    X_test_array = ts_fe.transform(X_test_ds)
    if not(X_comp is None):
        X_test_array = np.vstack((X_test_array, X_comp))
    # Regression
    y_pred_array = reg.predict(X_test_array)
    return y_pred_array


if __name__=="__main__":
    assignment = sys.argv[1]
    X_train_ds, y_train_array, X_test_ds, y_test_array = get_train_test_data(assignment)
    train_scores = []
    valid_scores = []
    test_scores = []
    X_comp, y_comp = get_compl_data(assignment, list_comp)
    len_comp = len(y_comp)
    for number, (train_is, valid_is) in enumerate(get_cv(y_train_array)):
        
        limit_comp = len_comp/5*(number+1)
        trained_model = train_submission('.', X_train_ds, y_train_array, train_is, X_comp[::limit_comp], y_comp[::limit_comp])
        
        y_train_pred_array = test_submission(trained_model, X_train_ds, train_is, X_comp[::limit_comp])

        train_score = score(
            np.concatenate((y_train_array[train_is].values[n_burn_in::], y_comp[::limit_comp])), y_train_pred_array)

        y_valid_pred_array = test_submission(
            trained_model, X_train_ds, valid_is)
        valid_score = score(y_train_array[valid_is].values[n_burn_in::], y_valid_pred_array)


        y_test_pred_array = test_submission(
            trained_model, X_test_ds, range(len(y_test_array)), X_comp[limit_comp::])
        test_score = score(np.concatenate((y_test_array[n_burn_in::], y_comp[limit_comp::])), y_test_pred_array)

        print('train RMSE = %s; valid RMSE = %s; test RMSE = %s' %
              (round(train_score, 3), round(valid_score, 3), round(test_score, 3)))

        train_scores.append(train_score)
        valid_scores.append(valid_score)
        test_scores.append(test_score)

    print(u'mean train RMSE = %s ± %s' %
          (round(np.mean(train_scores), 3), round(np.std(train_scores), 4)))
    print('mean valid RMSE = %s ± %s' %
          (round(np.mean(valid_scores[:-1]), 3), round(np.std(valid_scores), 4)))
    print('mean test RMSE = %s ± %s' %
          (round(np.mean(test_scores), 3), round(np.std(test_scores), 4)))
