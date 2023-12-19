
# README

This repository contains the source code for running the experiments of the master's thesis *Robustness Studies and Training Set Analysis for HIDS*. 
It bases on the Leipzig Intrusion Detection Data-Set ([LID-DS](https://github.com/LID-DS/LID-DS)). The original README of the LID-DS is also [included](README_LID-DS.md) in this repository. 
Besides being a dataset for evaluating anomaly-based HIDS, the LID-DS also offers an accompanying python framework, which is used and extended by this project. 

## Installation

Install python (at least 3.8) and install the requirements with:

```
pip install -r requirements.txt
```

## CLI Overview

In order to run and evaluate the experiments for the thesis, this reposistory offers a command-line tool in `cli.py`. It offers several sub-commands, which can be printed with `python cli.py --help`. 

```sh
usage: cli.py [-h]
              {run,check,tsa-dl,tsa-cv,tsa-combine,search,eval,tsa-augment,tsa-fs,tsa-ruleminer,tsa-correlate,tsa-add-suffix,tsa-stats,tsa-ngram-auc,tsa-concat,tsa-eval-fs,tsa-find-threshold}
              ...

positional arguments:
  {run,check,tsa-dl,tsa-cv,tsa-combine,search,eval,tsa-augment,tsa-fs,tsa-ruleminer,tsa-correlate,tsa-add-suffix,tsa-stats,tsa-ngram-auc,tsa-concat,tsa-eval-fs,tsa-find-threshold}
    run                 run experiment
    check               check experiment for completeness and integrity
    tsa-dl              analyse training set experiments
    tsa-cv              cross validate performance predictor
    tsa-combine         combine downloaded training set statistics and
                        performance measures
    search              run parameter search
    eval                evaluate multiple experiments for their robustness
    tsa-augment         analyse training set experiments
    tsa-fs              perform feature selection for tsa-cv
    tsa-ruleminer       get rules from performance predictor
    tsa-correlate       show correlations of training set statistics
    tsa-add-suffix      add suffix to features in csv file
    tsa-stats           print stats for training set characteristics
    tsa-ngram-auc       calculate Area-Under-Curve Values for n-gram related
                        metrics
    tsa-concat          concat training set characteristics data files
    tsa-eval-fs         evaluate feature selection results
    tsa-find-threshold  print stats for training set characteristics

optional arguments:
  -h, --help            show this help message and exit

```

## Reproducing Experiments

The following shows how the results of the master's thesis can be reproduced.

#### Setup 

First, mlflow must be setup. It can either be installed locally, or an instance at DataBricks may be utilized. Refer to [mlflow's documentation](https://mlflow.org/docs/latest/getting-started/index.html) for this. In case that DataBricks is used, it must be setup separately.

Furthermore, the experiments can be configured globally with a couple of environment variables. They should be set accordingly:

```sh

export EXPERIMENT_BASE_PATH=$(readlink -f experiments/slurm/)
export MLFLOW_TRACKING_URI="" # "" to use a local instance or "databricks" to use databricks (refer to the mlflow documentation)
export LID_DS_BASE="..." # The path to the LID-DS data
export EXPERIMENT_PREFIX="" # this can be left empty if mlflow is installed locally. If an mlflow instance at DataBricks is used, the prefix must match with the location of the experiments there.  
```

Furthermore, some experiments use caches for storing artifacts that can be reused by other experiment runs, or in order to resume an experiment run in case of being aborted. The cache files are stored in directories that must be set with the following environment variables:

```sh
export W2V_CACHE_PATH="path/to/cache" # used to store word2vec models 
export IDS_CACHE_PATH="path/to/cache" # used to store trained IDS objects (sufficient space is required)
export CACHE_PATH="path/to/cache" # used by various building blocks to cache internal data
```

It is possible to use the same directory for all caches. 

#### Robustness Experiments

The experiment configurations are stored in the directory `experiments/slurm/` as yaml files. For example, for reproducing the experiment for measuring the robustness of the STIDE baseline, the following command can be executed:

```sh
python cli.py run --config experiments/slurm/baseline/stide.yaml -e baseline-stide.yaml --continue random
```

The `-e` option sets the experiment name in mlflow. It can be chosen freely. However, if the name convention as above is utilized, i.e. the experiment name corresponds with the config path (by replacing `/` with `-` and using `$EXPERIMENT_BASE_PATH` as the root for this), the following commands can automatically infer the mlflow experiment name from the config. 

This command can be run multiple times in parallel. Each execution checks which experiment runs are still missing at mlflow and chooses a free run randomly. Note that according to the configuration at `experiments/slurm/baseline/stide.yaml`, the baseline STIDE experiment consists of over 400 runs.

To check the status of an experiment at mlflow, the following command can be used:

```sh
❯ python cli.py check --config experiments/slurm/baseline/stide.yaml -e baseline-stide.yaml
===> parameter_cfg_id: stide.yaml
Ignore scenarios: ['LID-DS-2021/CVE-2017-12635_6']
Progress: 0/414 (0.0 perc.) finished; 0/414 (0.0 perc.) running (out of missing runs)
NOT OK: 414 missing runs; 
====================================================
```

The `check` subcommand can remove stale or duplicate runs with `--remove-duplicate` or `--remove-stale` accordingly.

In order to evaluate a finished experiment, the following command can be used:

```sh
python cli.py eval --config experiments/slurm/baseline/stide.yaml --cache results/mlflow_cache --artifacts-dir results/baseline-stide
```

With this, the mlflow results are cached locally in the directory `results/mlflow_cache`, which must be created first. The diagrams and evaluation metrics are written to the artifacts directory `results/baseline-stide`.

The following experiment configurations are utilized in the thesis (in the directory `experiments/slurm`):

* Baselines (RSQ1.2): [experiments/slurm/baseline/stide.yaml](baseline/stide.yaml), `baseline/scg.yaml`, `baseline/som.yaml`
* Baseline with Combined Scenarios (RSQ1.2, Different Attacks): `combined/baseline/*`
* Baseline with Combined Scenarios (RSQ1.2, Same Attacks): `combined/baseline-same-attacks/*`
* 

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


# Optional: For post-processing some of the experiment results, [a command-line interface](https://github.com/dhelmr/pd) to the `pandas` python library is used, which must be installed separately. This is needed for generating latex tables from the results.
