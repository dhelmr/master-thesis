import itertools
import math
import pprint
import random
import statistics
import sys

import numpy as np
import pandas as pd
import sklearn.svm
from sklearn.decomposition import PCA
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier

from tsa.confusion_matrix import ConfusionMatrix

regressors = [
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
            df[col + "-n%s" % val] = new_col_values
    df = df.drop(columns=[drop_col] + modify_cols)
    df = df.drop_duplicates()
    df = df.reset_index()
    return df


class Heuristics1:
    def fit(self, *args, **kwargs):
        pass

    def predict(self, df):
        # preds = df["entropy"] * math.e / 10 - 1000 * df["u/t"]
        df["preds"] = df["entropy"] * math.e / 10 + df["u/t"] * (-1000)
        preds = [0.6 for i in range(len(df))]
        return preds


def train_test(train, test):
    drop_cols = ["f1_cfa", "syscalls", "num_attacks", "scenario"]
    Y_train = train["f1_cfa"]
    X_train = train.drop(columns=drop_cols)
    Y_test = np.array(test["f1_cfa"])
    X_test = test.drop(columns=drop_cols)
    scaler = MinMaxScaler()
    prepr_pl = Pipeline([
        ("min-max", MinMaxScaler()),
        # ("standard", StandardScaler()),
        # ("pca", PCA(n_components=50))
    ])
    # clf = SVC(kernel="rbf")
    clf = DecisionTreeClassifier(random_state=1)
    #clf = MLPClassifier(solver='adam', alpha=1e-5, max_iter=500,
    #             hidden_layer_sizes=(200, 10, 2), random_state=1)
    # clf = LinearRegression()
    # clf = Heuristics1()
    X_train_scaled = prepr_pl.fit_transform(X_train)
    # X_train_scaled = X_train
    clf.fit(X_train_scaled, Y_train)

    # print(clf.coef_, X_train.columns)
    # X_test_scaled = X_test
    X_test_scaled = prepr_pl.transform(X_test)
    preds = clf.predict(X_test_scaled)
    for i, p in enumerate(preds):
        if p < 0:
            preds[i] = 0
        if p > 1:
            preds[i] = 1
        # print(preds[i], Y_test[i])
    #err = mean_absolute_error(Y_test, preds)
    #err2 = mean_squared_error(Y_test, preds)
    cm = ConfusionMatrix.from_predictions(preds, Y_test, labels=[0,1])
    return cm


def main():
    df = pd.read_csv(sys.argv[1])
    df = df.drop(columns=["Unnamed: 0"])
    df["f1_cfa"] = df["f1_cfa"].apply(lambda f1: 1 if f1 > 0.75 else 0)

    aggregated_results = []
   # features = ["conditional_entropy", "entropy", "u/t", "unique", "simpson_index"]
    #features = ["unique_ngrams/n_threads", "unique_ngrams/total", "n_threads", "ngram_dists-simpson-index_mean", "ngram_dists_norm_entropy_mean", "nXt_pca-pca-evr-sum"]
    features = ["tXn_pca-pca-ev-n0", "unique_ngrams/total", "tXn_pca-pca-evr-n0", "tXn_pca-pca-noise-var", "nXt_pca-pca-evr-sum", "ngramXthreads_iod", "nXt_pca-pca-noise-var", "thread_dists_norm_entropy_mean"]
    for ss_l in range(len(features)-1):
        for subset in itertools.combinations(features, ss_l):
            if len(subset) == len(features):
                continue
            results = start_exp_run(df.drop(columns=[c for c in subset]))
            keep = [c for c in features if c not in subset]
            print("with: ", keep)
            for r in results:
                aggregated_results.append({
                    "query": r[0],
                    "features": ",".join(keep),
                    "f1": r[1],
                    "precision": r[2]
                })
    results_df = pd.DataFrame(aggregated_results)
    results_df.to_csv("classification_results.csv")

def start_exp_run(df):
    results = []
    for limit in range(1, 12):
        q = "ngram_length <= %s" % limit
        f1, prec = start_exp(unpack(df.query(q)))
        results.append((q, f1, prec))

        q = "ngram_length == %s" % limit
        df_filtered = df.query(q)
        f1, prec = start_exp(df_filtered.drop(columns=["ngram_length"]))
        results.append((q, f1, prec))
    sorted_f1 = sorted(results, key = lambda x: x[1], reverse=True)
    sorted_precision = sorted(results, key=lambda x: x[2], reverse=True)
    pprint.pprint(sorted_f1)
    pprint.pprint(sorted_precision)
    return results

def start_exp(df):
    scenarios = pd.unique(df["scenario"])
    f1 = []
    precs = []
    for test_sc in itertools.combinations(scenarios, 2):
        train = df[~df["scenario"].isin(test_sc)]  # df.query("scenario not in '%s'" % str(test_sc))
        test = df[df["scenario"].isin(test_sc)]  # df.query("scenario in '%s'" % str(test_sc))
        cm = train_test(train, test)
        metrics = cm.calc_unweighted_measurements()
        if metrics["f1_score"] >= 0 and metrics["f1_score"] <= 1:
            f1.append(metrics["f1_score"])
        else:
            # print("warning: invalid f1_score", cm, metrics["f1_score"])
            f1.append(0)
        if metrics["precision"] >= 0 and metrics["precision"] <= 1:
            precs.append(metrics["precision"])
        else:
            # print("invalid precision", metrics["precision"])
            precs.append(0)

    return statistics.mean(f1), statistics.mean(precs)



if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    main()
