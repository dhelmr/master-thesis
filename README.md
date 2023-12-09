
# README

This repository contains the source code for running the experiments of the master's thesis **TODO**. It bases on the Leipzig Intrusion Detection Data-Set ([LID-DS](https://github.com/LID-DS/LID-DS)). The original README of the LID-DS is also [included](README_LID-DS.md) in this repository. 
Besides being a dataset for evaluating anomaly-based HIDS, the LID-DS also offers an accompanying python framework, which is used by this project. 

## Installation

TODO

For post-processing some of the experiment results, [a command-line interface](https://github.com/dhelmr/pd) to the `pandas` python library is used, which must be installed separately. This is needed for generating latex tables from the results.

### Setup MLflow

TODO

## CLI Overview

In order to run and evaluate the experiments for the thesis, this reposistory offers a command-line tool in `cli.py`. It offers several sub-commands, which can be printed with `python cli.py --help`. 

TODO: kurze Erklärung der einzelnen sub-commands



## Reproducing Experiments

The following shows how the results of the master's thesis can be reproduced.

#### Training Set Suitability Estimation

## Create csv file containing training set statistics and performance metrics

This assumes that the experiments in `experiments/slurm/analysis/all-ngrams.yaml` and `experiments/slurm/analysis/stide/*` are already completed in mlflow. 

```sh
# set EXPERIMENT_PREFIX according to mlflow tracking server
export EXPERIMENT_PREFIX="..."
# download training set statistics
python cli.py tsa-dl-e $EXPERIMENT_PREFIX/analysis-all-ngrams.yaml --config experiments/slurm/analysis/all-ngrams.yaml -o test-all-ngrams.csv
# combine f1/prec/recall performance stats with training set statistics
python cli.py tsa-combine --statistics-csv test-all-ngrams.csv -e $EXPERIMENT_PREFIX/analysis-stide-max_syscalls.search.yaml -o test-perf-all-ngrams.csv 
```

---
