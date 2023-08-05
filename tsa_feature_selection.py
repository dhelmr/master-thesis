import argparse
import itertools
import math
import pprint
import random
import statistics
import sys
from typing import List

import numpy as np
import pandas as pd
import sklearn.svm
from sklearn.tree import export_text
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeClassifier
from tqdm import tqdm

from tsa.confusion_matrix import ConfusionMatrix

AVAILABLE_METRICS = ["f1_cfa"]
KNOWN_NON_FEATURE_COLS = ["scenario", "syscalls"]

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", required=True)
parser.add_argument("--metric", default="f1_cfa")
parser.add_argument("--threshold", type=float, default=0.8)
parser.add_argument("--cv-leave-out", type=int, default=2)
parser.add_argument("--out", required=True)
parser.add_argument("--start-features", nargs="+", default=[])
parser.add_argument("--max-depth", default=10, type=int)
parser.add_argument("--iterations", default=10, type=int)
args = parser.parse_args()


def train_test(train, test):
    # TODO change metric
    drop_cols = ["f1_cfa", "scenario"]
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
    clf = DecisionTreeClassifier(random_state=1, max_depth=args.max_depth)
    # clf = RandomForestClassifier(random_state=1)
    # clf = MLPClassifier(solver='adam', alpha=1e-5, max_iter=500,
    #             hidden_layer_sizes=(200, 10, 2), random_state=1)
    # clf = LogisticRegression()
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
    # err = mean_absolute_error(Y_test, preds)
    # err2 = mean_squared_error(Y_test, preds)
    cm = ConfusionMatrix.from_predictions(preds, Y_test, labels=[0, 1])
    #print(export_text(clf))
    return cm


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns
    drop_cols = [c for c in cols if "." in c
                 or c in ["permutation_i", "parameter_cfg_id", "iteration", "run_id", "num_attacks", "syscalls"]
                 or str(c).startswith("Unnamed")]
    return df.drop(columns=drop_cols)


def get_features(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if col not in KNOWN_NON_FEATURE_COLS + AVAILABLE_METRICS]


def main():
    df = pd.read_csv(args.input)
    df = clean_dataset(df)

    df[args.metric] = df[args.metric].apply(lambda f1: 1 if f1 > args.threshold else 0)
    available_features = set(get_features(df))
    selected_features = args.start_features

    for f in selected_features:
        available_features.remove(f)

    aggregated_results = []
    skip_features = []

    i = 0
    while len(available_features) > 0:
        print("Start run %s" % i)
        print("selected features", selected_features)
        run_results = []
        f_to_remove = []

        current_best = None

        pbar = tqdm(available_features)
        for f in pbar:
            feature_set = selected_features + [f]
            drop_cols = [c for c in available_features if c != f] + skip_features
            try:
                results = start_exp_run(df.drop(columns=drop_cols), leave_out=args.cv_leave_out)[0]
                run_results.append((f, results))
                aggregated_results.append({
                    "query": results[0],
                    "features": ",".join(feature_set),
                    "f1": results[1],
                    "precision": results[2]
                })
                if current_best is None or current_best[1] < results[1]:
                    current_best = results + (f,)
                pbar.set_description("Current best: %s" % (current_best,))
            except Exception as e:
                print(e)
                print("Remove %s from feature set permanently" % f)
                f_to_remove.append(f)
        for f in f_to_remove:
            available_features.remove(f)
            skip_features.append(f)
        run_results.sort(key=lambda r: r[1][1], reverse=True)
        best_feature = run_results[0][0]
        print("best feature of run %s: %s (results: %s)" % (i, best_feature, run_results[0][1]))
        selected_features.append(best_feature)
        available_features.remove(best_feature)
        i += 1
        if i >= args.iterations:
            print("Finished")
            break
    results_df = pd.DataFrame(aggregated_results)
    results_df.to_csv(args.out)


def start_exp_run(df, leave_out):
    results = []
    f1, pr = start_exp(df, leave_out)
    results.append(("", f1, pr))
    sorted_f1 = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_precision = sorted(results, key=lambda x: x[2], reverse=True)
    return results


def start_exp(df, leave_out=2):
    scenarios = pd.unique(df["scenario"])
    f1 = []
    precs = []
    for test_sc in itertools.combinations(scenarios, leave_out):
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
