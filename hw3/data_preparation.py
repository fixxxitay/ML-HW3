import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.under_sampling import NearMiss
from collections import Counter

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from hw3.features import *

right_feature_set = ["Vote", "Yearly_IncomeK", "Number_of_differnt_parties_voted_for", "Political_interest_Total_Score",
                     "Avg_Satisfaction_with_previous_vote", "Avg_monthly_income_all_years",
                     "Most_Important_Issue", "Overall_happiness_score", "Avg_size_per_room",
                     "Weighted_education_rank"]


def deterministic_split(df, train, test):
    df_train = df.iloc[0:round(len(df) * train), :]
    df_test = df.iloc[round(len(df) * train):round(len(df) * (train + test)), :]
    df_validation = df.iloc[round(len(df) * (train + test)):len(df), :]

    return df_train, df_test, df_validation


def balanced_split(df, train):
    nr = NearMiss()
    x_train = train[:, 1:]
    y_train = train[:, 1]
    x_train_miss, y_train_miss = nr.fit_sample(x_train, y_train)

    return x_train_miss, y_train_miss


def save_files(df_train, df_test, df_validation):
    df_train.to_csv('prepared_train.csv', index=False)
    df_validation.to_csv('prepared_validation.csv', index=False)
    df_test.to_csv('prepared_test.csv', index=False)


def remove_wrong_party_and_na(df_train, df_test, df_validation):
    df_train = df_train[df_train.Vote != 10]
    df_train = df_train[df_train.Vote != 4]
    df_train = df_train.dropna()

    df_test = df_test[df_test.Vote != 10]
    df_test = df_test[df_test.Vote != 4]
    df_test = df_test.dropna()

    df_validation = df_validation[df_validation.Vote != 10]
    df_validation = df_validation[df_validation.Vote != 4]
    df_validation = df_validation.dropna()

    return df_train, df_test, df_validation


def save_raw_data(df_test, df_train, df_validation):
    df_train.to_csv('raw_train.csv', index=False)
    df_test.to_csv('raw_test.csv', index=False)
    df_validation.to_csv('raw_validation.csv', index=False)


def complete_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame, df_validation: pd.DataFrame) -> (
        pd.DataFrame, pd.DataFrame, pd.DataFrame):
    df_train = df_train[df_train > 0]
    df_test = df_test[df_test > 0]
    df_validation = df_validation[df_validation > 0]

    for col in df_train.columns.values:
        if col == 'Vote':
            df_train[col].fillna(df_train[col].mode()[0], inplace=True)
            continue

        filler = None
        if col in nominal_features:
            filler = df_train[col].mode()[0]

        if col in integer_features:
            filler = round(df_train[col].mean())

        if col in float_features:
            filler = df_train[col].mean()

        df_train[col].fillna(filler, inplace=True)
        df_test[col].fillna(filler, inplace=True)
        df_validation[col].fillna(filler, inplace=True)

    return df_train, df_test, df_validation


def nominal_to_numerical_categories(df: pd.DataFrame):
    # from nominal to Categorical
    df = df.apply(lambda x: pd.Categorical(x) if x.dtype != 'float64' else x, axis=0)
    # give number to each Categorical
    df = df.apply(lambda x: x.cat.codes if x.dtype != 'float64' else x, axis=0)
    return df


def apply_feature_selection(df_train, df_test, df_validation, feature_set):
    df_train = df_train[feature_set]
    df_test = df_test[feature_set]
    df_validation = df_validation[feature_set]

    return df_train, df_test, df_validation


def normalize(df_test: pd.DataFrame, df_train: pd.DataFrame, df_validation: pd.DataFrame):
    # min-max for uniform features
    uniform_scaler = MinMaxScaler(feature_range=(-1, 1))
    df_train[uniform_features] = uniform_scaler.fit_transform(df_train[uniform_features])
    df_validation[uniform_features] = uniform_scaler.transform(df_validation[uniform_features])
    df_test[uniform_features] = uniform_scaler.transform(df_test[uniform_features])

    # z-score for normal features
    normal_scaler = StandardScaler()
    df_train[normal_features] = normal_scaler.fit_transform(df_train[normal_features])
    df_validation[normal_features] = normal_scaler.transform(df_validation[normal_features])
    df_test[normal_features] = normal_scaler.transform(df_test[normal_features])
    return df_train, df_test, df_validation


def remove_outliers(threshold: float, df_train: pd.DataFrame, df_validation: pd.DataFrame, df_test: pd.DataFrame):
    mean = df_train[normal_features].mean()
    std = df_train[normal_features].std()

    z_train = (df_train[normal_features] - mean) / std
    z_val = (df_validation[normal_features] - mean) / std
    z_test = (df_test[normal_features] - mean) / std

    df_train[z_train.mask(abs(z_train) > threshold).isna()] = np.nan
    df_validation[z_val.mask(abs(z_val) > threshold).isna()] = np.nan
    df_test[z_test.mask(abs(z_test) > threshold).isna()] = np.nan

    return df_train, df_validation, df_test


def main():
    # first part - data preparation
    # step number 1
    # Load the Election Challenge data from the ElectionsData.csv file
    """
    1. Can be found at the “The Election Challenge” section in the course site
    2. Make sure that you’ve identified the correct type of each attribute
    """
    df = pd.read_csv("ElectionsData.csv")

    # step number 2
    # Select the right set of features (as listed in the end of this document),
    # and apply the data preparation tasks that you’ve carried out in the former
    # exercise, on the train, validation, and test data sets
    """
    1. At the very least, handle fill up missing values
        All other actions, such as outlier detection and normalization,
        are not mandatory. Still, you are encouraged to do them,
        as they should provide you added values for the modeling part
    2. Whether to use the validation set for pre-processing steps is your call
    """
    # Convert nominal types to numerical categories
    df = nominal_to_numerical_categories(df)

    # split the data to train , test and validation
    df_train, df_test, df_validation = deterministic_split(df, 0.6, 0.2)

    # Save the raw data first
    save_raw_data(df_test, df_train, df_validation)

    # 1 - Imputation - Complete missing values
    df_train, df_test, df_validation = complete_missing_values(df_train, df_test, df_validation)

    # 2 - Data Cleansing
    # Outlier detection using z score
    threshold = 3  # .3
    df_train, df_validation, df_test = remove_outliers(threshold, df_train, df_validation, df_test)

    # Remove lines with wrong party (Violets | Khakis)
    df_train, df_test, df_validation = remove_wrong_party_and_na(df_train, df_test, df_validation)

    # 3 - Normalization (scaling)
    df_train, df_test, df_validation = normalize(df_test, df_train, df_validation)

    # apply feature selection
    feature_set = right_feature_set
    df_train, df_test, df_validation = apply_feature_selection(df_train, df_test, df_validation, feature_set)

    print(df_train)
    print(df_test)
    print(df_validation)
    print(df_train["Vote"])
    counter = Counter(df_train["Vote"])
    print(counter)

    nr = NearMiss()
    x_train = df_train.drop("Vote", 1)
    y_train = df_train["Vote"]
    


    # x_train_miss, y_train_miss = nr.fit_sample(x_train, y_train)
    # print(x_train_miss)
    # print("#############################################")
    # print(y_train_miss)

    # step number 3
    # Save the 3x2 data sets in CSV files
    # CSV files of the prepared train, validation and test data sets
    # save_files(df_train, df_test, df_validation)


if __name__ == '__main__':
    main()
