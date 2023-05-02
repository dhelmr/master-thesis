import argparse
import mlflow
from mlflow import MlflowClient

import matplotlib.pyplot as plt

def evaluate_exp(exp_id: str, ax_f1, ax_precision, ax_recall):
    exp = client.get_experiment(exp_id)
    print(exp.name)
    runs = mlflow.search_runs(experiment_ids=[exp_id])
    print("grouped by scenario")
    print(runs.groupby(by="params.scenario").mean()[["metrics.ids.f1_cfa", "metrics.ids.precision_with_cfa", "metrics.ids.recall"]])
    grouped_attacks = runs.groupby(by="params.num_attacks").mean()[["metrics.ids.f1_cfa", "metrics.ids.precision_with_cfa", "metrics.ids.recall"]]
    print(grouped_attacks)

    grouped_attacks.plot(y="metrics.ids.f1_cfa", label="f1 "+exp.name, ax=ax_f1)
    grouped_attacks.plot(y="metrics.ids.precision_with_cfa", label="precision "+exp.name, ax=ax_precision)
    grouped_attacks.plot(y="metrics.ids.recall", label="recall " + exp.name, ax=ax_recall)

def create_axes(n):
    axes = []
    for i in range(n):
        _, ax = plt.subplots(1)
        ax.set_ylim(ymin=0, ymax=1)
        axes.append(ax)
    return tuple(axes)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", "-e", required=True, nargs="+")
    #parser.add_argument("--scenario_name", "-s", required=True)
    parsed = parser.parse_args()

    client = MlflowClient()
    ax_f1, ax_pr, ax_recall = create_axes(3)
    for e_id in parsed.experiment_id:
        results = evaluate_exp(e_id, ax_f1, ax_pr, ax_recall)
        print(results)
    plt.show()