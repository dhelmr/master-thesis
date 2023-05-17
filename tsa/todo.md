
# Mai II (16-21)

## experiment management

* dataloader permutation_i as list
* output IDS block parameters in mlflow params
* refactor experiment to ParameterCfg
    - mlflow: parameter_cfg_id
    - ExperimentChecker: check for parameter_cfg_id+iteration
* global config => store experiment prefix => make -e param optional (infer from config name)

## Advanced Dataloder Features

* load attacks from different scenario + Combine Scenarios 
* Specify Number of syscalls Loaded 
* Load more Data until certain Measure (Entropy, ... (?) ) is reached

## TSA


* LOFAnalyser => make use of local outlier factor values (mean, variance, ...)
* ClusteringAnalyser => cluster-wise entropy?
* entropy values from paper
* model zipf distribution characteristics

## Experiment Evaluation

* plot total number of false alarms

---

# Ideas/Backlog

* Training Set Visualization
* Zipf Law for Training Set?
* dimensions of features vs number of needed training instances?
* date exploration with apache superset, etc.

## Code Maintainance / SWE

* problem: permutation_i = 0 == permutation_i = 1 (nur andere Reihenfolge)
* backup databricks mlflow runs
* use jupyter in slurm
* Feature Cache

## performance

* Serialize W2V trained model per training dataset cfg in sqlite

## Robustness

* Check MixedModel implementation
* Add Robust AE implementation as DE

## TSA Visualization

* t-sne with ngram distance methods

## Experiments



---


# May I (09-15)



## More Scitkit-Learn Algorithms

* Elliptic Curve
* OCSVM
* Robust Covariance
* 

## experiments

* LOF
    - contamination auto
    - n neighbors
    - unique
* T-Stide
    - ngram length
    - threshold
    - thread_aware
    - permutation_i 
* Stide/SCG shorter SUM window
* Stide/SCG/SOM: permutation_i = 10

---
