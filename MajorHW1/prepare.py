from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import pandas as pd

def prepare_data(training_data, new_data):
  # make a copy of new_data
  temp_data = new_data.copy()

  # fill null entries in household incomde with the median
  temp_data.household_income = temp_data.household_income.fillna(training_data.household_income.median())

  # create SpecialProperty feature
  if 'SpecialProperty' not in temp_data.columns:
    newGroup1 = temp_data["blood_type"].isin(["O+", "B+"])
    temp_data.insert(2, "SpecialProperty", newGroup1, True)

  # remove bloot type column
  temp_data = temp_data.drop(columns=['blood_type'])

  # scale features according to the train data
  standard_scaler= StandardScaler()
  min_max_scaler = MinMaxScaler(feature_range=(-1,1))
  features_standard = ['PCR_01', 'PCR_02', 'PCR_05', 'PCR_06', 'PCR_07', 'PCR_08']
  features_min_max = ['PCR_03', 'PCR_04', 'PCR_09', 'PCR_10']

  # fit the scalers to the training data statistics
  standard_scaler.fit(training_data[features_standard])
  min_max_scaler.fit(training_data[features_min_max])

  # apply the scalers on the relevant features
  temp_data[features_standard] = standard_scaler.transform(temp_data[features_standard])
  temp_data[features_min_max] = min_max_scaler.transform(temp_data[features_min_max])

  return temp_data