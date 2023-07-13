import itertools
import math
import random
import statistics

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR

classifier = [
    MLPRegressor(solver='adam', alpha=1e-5, max_iter=500,
                     hidden_layer_sizes=(200, 100, 80, 40, 40, 10, 2), random_state=1),
    LinearRegression(),
    Lasso(),
    RidgeCV(),
    ElasticNet(),
    SVR(),
    KernelRidge()
]


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

class Heuristics1:
    def fit(self, *args, **kwargs):
        pass

    def predict(self, df):
        #preds = df["entropy"] * math.e / 10 - 1000 * df["u/t"]
        df["preds"] = df["entropy"] * math.e / 10 + df["u/t"]*(-1000)
        preds = [0.6 for i in range(len(df))]
        return preds

def train_test(train, test):
    drop_cols = ["f1_cfa", "syscalls", "num_attacks", "scenario"]
    Y_train = train["f1_cfa"]
    X_train = train.drop(columns = drop_cols)
    Y_test = np.array(test["f1_cfa"])
    X_test = test.drop(columns = drop_cols)
    scaler = MinMaxScaler()
    prepr_pl = Pipeline([
        ("min-max", MinMaxScaler()),
        #("standard", StandardScaler()),
        #("pca", PCA(n_components=50))
    ])
    clf = SVR(kernel="rbf")
   # clf = LinearRegression()
    #clf = Heuristics1()
    X_train_scaled = prepr_pl.fit_transform(X_train)
   # X_train_scaled = X_train
    clf.fit(X_train_scaled, Y_train)

   # print(clf.coef_, X_train.columns)
    #X_test_scaled = X_test
    X_test_scaled = prepr_pl.transform(X_test)
    preds = clf.predict(X_test_scaled)
    for i, p in enumerate(preds):
        if p < 0:
            preds[i] = 0
        if p > 1:
            preds[i] = 1
        #print(preds[i], Y_test[i])
    err = mean_absolute_error(Y_test, preds)
    err2 = mean_squared_error(Y_test, preds)
    return err, err2


def main():
    df = pd.read_csv("entropyXsyscalls.csv")
    df = df.drop(columns=["Unnamed: 0"])


    df.drop(columns=["unique", "conditional_entropy", "simpson_index"], inplace=True)
    for limit in range(1,15):
        print("<= limit", limit)
        start_exp(unpack(df.query("ngram_length <= %s" % limit)))
        print("only = limit", limit)
        df_filtered = df.query("ngram_length == %s" % limit )
        start_exp(df_filtered.drop(columns=["ngram_length"]))

def start_exp(df):

    scenarios = pd.unique(df["scenario"])
    errs = []
    errs2 = []
    for test_sc in itertools.combinations(scenarios, 2):
        train = df[~df["scenario"].isin(test_sc)] # df.query("scenario not in '%s'" % str(test_sc))
        test = df[df["scenario"].isin(test_sc)] #df.query("scenario in '%s'" % str(test_sc))
        err, err2 = train_test(train, test)
        errs.append(err)
        errs2.append(err2)
    print("mean errors:")
    print(statistics.mean(errs), statistics.mean(errs2))

if __name__ == '__main__':
    main()

