import statistics

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def unpack(df: pd.DataFrame):
    index_cols = ["scenario", "syscalls", "num_attacks"]
    val_col = ["f1_cfa"]
    drop_col = "ngram_length"
    append_values = pd.unique(df[drop_col])
    df = df.set_index(index_cols, drop=True)
    modify_cols = [col for col in df.columns if col not in index_cols + val_col + [drop_col]]
    for val in append_values:
        for col in modify_cols:
            new_col_values = df[col][df[drop_col] == val]
            df[col+"-n%s" % val] = new_col_values
    df = df.drop(columns=[drop_col]+modify_cols)
    df = df.drop_duplicates()
    df = df.reset_index()
    return df
def train_test(train, test):
    Y_train = train["f1_cfa"]
    X_train = train.drop(columns = ["f1_cfa", "syscalls", "num_attacks", "scenario"])
    Y_test = test["f1_cfa"]
    X_test = test.drop(columns = ["f1_cfa", "syscalls", "num_attacks", "scenario"])
    scaler = MinMaxScaler()
    clf = MLPRegressor(solver='adam', alpha=1e-5, max_iter=500,
                         hidden_layer_sizes=(200, 100, 80, 40, 40, 10, 2), random_state=1)
    X_train_scaled = scaler.fit_transform(X_train)
    clf.fit(X_train_scaled, Y_train)
    preds = clf.predict(scaler.transform(X_test))
    for i, p in enumerate(preds):
        if p < 0:
            preds[i] = 0
        if p > 1:
            preds[i] = 1
    print(preds)
    err = mean_absolute_error(Y_test, preds)
    err2 = mean_squared_error(Y_test, preds)
    return err, err2


def main():
    df = pd.read_csv("entropyXsyscalls.csv")
    df = df.drop(columns=["Unnamed: 0"])

    scenarios = pd.unique(df["scenario"])
    df = unpack(df)

    errs = []
    errs2 = []
    for test_sc in scenarios:
        train = df.query("scenario != '%s'" % test_sc)
        test = df.query("scenario == '%s'" % test_sc)
        err, err2 = train_test(train, test)
        errs.append(err)
        errs2.append(err2)
    print(statistics.mean(errs), statistics.mean(errs2))

if __name__ == '__main__':
    main()

