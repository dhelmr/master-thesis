
# README

This repository contains the source code for running the experiments of the master's thesis *Robustness Studies and Training Set Analysis for HIDS*. 
It bases on the Leipzig Intrusion Detection Data-Set ([LID-DS](https://github.com/LID-DS/LID-DS)). The original README of the LID-DS is also [included](README_LID-DS.md) in this repository. 
Besides being a dataset for evaluating anomaly-based HIDS, the LID-DS also offers an accompanying python framework, which is used and extended by this project. The remainder of this README first gives a short overview about the command-line interface to run the experiments, and about the source code contributed by the thesis. Then, it outlines how the experiments can be reproduced.

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

---

## Source Code Overview

As stated above, this repository contains the Python framework provided by the LID-DS. It is included in the directories `algorithms/`, `tools/`, and `dataloader`. Note that this repository does *not* include the `scenario/` and `lid_ds` folders of the original LID-DS repository, since they are only needed for creating the dataset itself. For this thesis however, the already-available data of the existing LID-DS versions (2019 and 2021) is used. 

The main source code contribution of the thesis is contained in the directory `tsa/`:

* The module `tsa.cli` includes the implementation of the various subcommands of the `cli.py` command-line interface.
* The module `tsa.accommodation` includes the implementation of various accommodation appraoches for improving the robustness of HIDS.
* The module `tsa.diagnosis` includes the implementation of various diagnosis approaches for improving the robustness of HIDS. They are implemented as building blocks that pre-process the dataset.
* The module `tsa.analysis` includes the implementation of common training set analysis techniques and building blocks for the suitability dataset creation.
* The module `tsa.dataloaders` includes the dataloader enhancements used by the experiments.
* The module `tsa.mlflow` includes code for the integration of mlflow.
* The module `tsa.perf_pred` includes code that is used for the suitability prediction in RSQ2.2 (e.g. the cross validation and decision trees)

Furthermore, the original LID-DS implementation of the SOM (`algorithms/decision_engines/som.py`) is modified, and the Word2Vec-Vectors are modified to ensure deterministic behavior. 

The directory `experiments/` contains the configuration files for all experiments utilizing the LID-DS. They are stored as yaml files in order to ensure reproducibility and transparency. The `test/` directory contains various unit tests for the added source code. The `scripts/` directory contains various scripts that are used to create the results shown in the thesis (e.g. diagrams and tables).

---

## Reproducing Experiments

The following shows how the results of the master's thesis can be reproduced.

#### Setup 

1. First, mlflow must be setup. It can either be installed locally, or an instance at DataBricks may be utilized. Refer to [mlflow's documentation](https://mlflow.org/docs/latest/getting-started/index.html) for this. In case that DataBricks is used, it must be setup separately.

2. The LID-DS dataset (both versions) must be downloaded (refer to the LID-DS README), e.g. to the directory `data/`.

3. Furthermore, the experiments can be configured globally with a couple of environment variables. They must be set accordingly:

```sh

export EXPERIMENT_BASE_PATH=$(readlink -f experiments/slurm/)
export MLFLOW_TRACKING_URI="" # "" to use a local instance or "databricks" to use databricks (refer to the mlflow documentation)
export LID_DS_BASE=$(readlink -f data) # The path to the LID-DS data
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

The following experiment configurations are utilized in the thesis (in the directory `experiments/slurm`) for research question 1.2:

* Baselines (RSQ1.2): [baseline/stide.yaml](experiments/slurm/baseline/stide.yaml), [baseline/scg.yaml](experiments/slurm/baseline/scg.yaml), [baseline/som.yaml](experiments/slurm/baseline/som.yaml)
* Baseline with Combined Scenarios (RSQ1.2, Different Attacks): [combined/baseline/*](experiments/slurm/combined/baseline)
* Baseline with Combined Scenarios (RSQ1.2, Same Attacks):  [combined/baseline-same-attacks/*](experiments/slurm/combined/baseline-same-attacks)

For the robustness improvement experiments (cf. section 6.2 in the thesis), the following configurations are used:

* "6.2.1 Influence of N-Gram Size on STIDE’s Robustness": `experiments/slurm/baseline/stide-n2.yaml experiments/slurm/baseline/stide-n3.yaml experiments/slurm/baseline/stide.yaml experiments/slurm/baseline/stide-n6.yaml experiments/slurm/baseline/stide-n7.yaml experiments/slurm/baseline/stide-n10.yaml experiments/slurm/baseline/stide-n15.yaml experiments/slurm/baseline/stide-n20.yaml`
* "6.2.2 f-STIDE": `experiments/slurm/f-stide/f-stide-exp-a0.3.yaml experiments/slurm/f-stide/f-stide-exp-a0.5.yaml experiments/slurm/f-stide/f-stide-exp-a0.7.yaml experiments/slurm/f-stide/f-stide-exp-a0.9.yaml experiments/slurm/f-stide/f-stide-exp-a0.95.yaml experiments/slurm/f-stide/f-stide-homographic-a0.5.yaml experiments/slurm/f-stide/f-stide-homographic-a2.yaml experiments/slurm/f-stide/f-stide-homographic-a5.yaml experiments/slurm/f-stide/f-stide-max-scaled.yaml`
* "6.2.3 Thread-Based STIDE Accommodation": `experiments/slurm/thread-f-stide/norm_entropy.yaml experiments/slurm/thread-f-stide/thread-freq-homographic-a1.yaml experiments/slurm/thread-f-stide/thread-freq-homographic-a2.yaml experiments/slurm/tfidf-stide/tfidf_stide-mean-1.yaml`
* "6.2.4 SCG Accommodation: Thread-Wise Graphs": ` experiments/slurm/baseline/scg{,-thread-wise-graphs}.yaml`
* "6.2.5 SOM Accommodation: Frequency Features": `experiments/slurm/frequency_append/som-{ngram,thread,ngram-thread}.yaml experiments/slurm/baseline/som.yaml`
* "6.2.6 SOM Accommodation: Smaller Sizes": `experiments/slurm/som/size0.{5,7,9}.yaml experiments/slurm/baseline/som.yaml`
* "6.2.7 SOM Accommodation: Frequency-Based Sampling": `experiments/slurm/som/sample-0.{5,7,9,95,99}.yaml experiments/slurm/baseline/som.yaml`
* "6.2.8 Combination of SOM Accommodation Methods": `experiments/slurm/som/sample-{0.95,0.99}.yaml experiments/slurm/som/sample-{0.95,0.99}+thread-size0.7.yaml`
* "6.2.9 Frequency-Based Outlier Diagnosis" (absolute thresholds): `experiments/slurm/preprocessing/frequency-od/stide-t1.yaml experiments/slurm/preprocessing/frequency-od/stide-t3.yaml experiments/slurm/preprocessing/frequency-od/stide-t8.yaml experiments/slurm/preprocessing/frequency-od/stide-t15.yaml`
* "6.2.9 Frequency-Based Outlier Diagnosis" (relative thresholds): `experiments/slurm/preprocessing/frequency-od/rel/stide-0.00005.yaml experiments/slurm/preprocessing/frequency-od/rel/stide-0.00001.yaml experiments/slurm/preprocessing/frequency-od/rel/stide-0.000005.yaml` 
* "6.2.10 Thread-Based Outlier Diagnosis" (no tfidf): `experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n1.yaml experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n3.yaml experiments/slurm/baseline/stide.yaml`
* "6.2.10 Thread-Based Outlier Diagnosis" (tfidf): `experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n1.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n3.yaml experiments/slurm/baseline/stide.yaml`
* "6.2.10 Thread-Based Outlier Diagnosis" (distances):  ` experiments/slurm/preprocessing/thread-od/lof-{binary-jaccard,jds}-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-hellinger-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jds-n2.yaml experiments/slurm/baseline/stide.yaml`
* "6.2.11 Combination of Thread-Based Outlier Diagnosis with SCG and STIDE
Accomodation" (Thread-OD + thread-f-stide): `experiments/slurm/preprocessing/thread-od/tfidf-lof-cosine-n2.yaml experiments/slurm/thread-f-stide/thread-freq-homographic-a2.yaml experiments/slurm/combination/stide/lof-cosine+thread-freq-homographic-a2.yaml`
* "6.2.11 Combination of Thread-Based Outlier Diagnosis with SCG and STIDE
Accomodation" (Thread-OD + norm-entropy-stide): `experiments/slurm/preprocessing/thread-od/tfidf-lof-cosine-n2.yaml experiments/slurm/thread-f-stide/norm_entropy.yaml experiments/slurm/combination/stide/lof-cosine+norm-entropy.yaml`
* "6.2.11 Combination of Thread-Based Outlier Diagnosis with SCG and STIDE
Accomodation" (Thread-OD + SCG+thread-wise graphs): `experiments/slurm/baseline/scg{,-thread-wise-graphs}.yaml experiments/slurm/combination/scg/lof-cosine+scg-baseline.yaml  experiments/slurm/combination/scg/lof-cosine+thread-wise.yaml`



#### Training Set Suitability Estimation

Note: For post-processing some of the experiment results, [a command-line interface](https://github.com/dhelmr/pd) to the `pandas` python library is used, which must be installed separately. It is needed for generating latex tables from the results.

##### Suitability Datasets

The experiments for creating the suitability datasets are:

* For the STIDE f1-score suitability quantifier: `max_syscalls-big.search.yaml`, `max_syscalls-medium.search.yaml`, and `max_syscalls-small.search.yaml` in `experiments/slurm/analysis/stide`
* for the general n-gram frequency statistics: `analysis/all-ngrams.yaml`
* For the thread-n-gram matrix statistics: `analysis/thread_matrix-n2.yaml` and `analysis/thread_matrix-n3.yaml`
* For the data drift suitability quantifiers: `analysis/data-drift-no-attacks.yaml`

Once they are finished in mlflow, they can then be aggregated and downloaded with:

```sh
python cli.py tsa-dl -o results/thread_matrix-n3.csv -e $EXPERIMENT_PREFIX/analysis-thread_matrix-n3.yaml --config experiments/slurm/analysis/thread_matrix-n3.yaml
python cli.py tsa-dl -o results/thread_matrix-n2.csv -e $EXPERIMENT_PREFIX/analysis-thread_matrix-n3.yaml --config experiments/slurm/analysis/thread_matrix-n2.yaml
python cli.py tsa-dl -o results/analysis-all-ngrams.csv -e $EXPERIMENT_PREFIX/analysis-all-ngrams.yaml --config experiments/slurm/analysis/all-ngrams.yaml
python cli.py tsa-dl -o results/data-drift-no-attacks.csv -e $EXPERIMENT_PREFIX/analysis-data_drift-data-drift-no-attacks.yaml --config experiments/slurm/analysis/data_drift/data-drift-no-attacks.yaml
```

##### RSQ1.1

The following scripts generate the results for RSQ2.1:

```sh
scripts/make_rsq2.1.sh # creates/augments the data
scripts/make_rsq2.1-corr.sh # calculates the correlation coefficients
scripts/make_rsq2.1-tables.sh # creates the latex tables used in the thesis
```

##### RSQ2.1

The following scripts generate the results for RSQ2.2:

```sh
scripts/make_rsq2.2.sh # creates/augments the data
scripts/make_rsq2.2-baselines.sh # calculates the baseline results for the datasets
scripts/make_rsq2.2-stats-tables.sh # Creates the class distribution tables used in the tables
scripts/make_rsq2.2-slurmjobs.sh # Runs the feature selection on SLURM
scripts/make_rsq2.2-results.sh # Selects the best features and generates the tables used in the thesis
```

##### RSQ2.3
The following scripts generate the results for RSQ2.3:

```sh
scripts/make_rsq2.3-dt.sh # generate decision tree SVGs
```

---

## LICENSE

See the file [LICENSE](LICENSE).