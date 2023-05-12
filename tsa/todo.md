## Code Maintainance / SWE

* output IDS block parameters in mlflow params
* easy way to run all runs of experiment in slurm parallel
* parameter search
* Refactor unsupervised.py
* backup databricks mlflow runs
* use jupyter in slurm
* Feature Cache

## Ideas/Backlog

* Training Set Visualization
* Zipf Law for Training Set?
* dimensions of features vs number of needed training instances?

# May I (09-15)

## Refactoring

* num_runs problem
* refactor + merge unsupervised eval
* slurm files in git

## slurm

* input yaml config => create experiment name and run automatically
* create parallel jobs

## experiment management

* output IDS block parameters in mlflow params
* implement parameter grid search 
* list missing runs in experiment
* check experiment integrity/completeness

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

## TSA

* new experiment mode for analysis
* total/unique features
* for density-based features: variance, ...
* entropy values from paper

## Training Set Visualization

* Visualize Building Block => PCA/tnsa dimension reduction

# Mai II (16-21)

## SWE 

* Serialize W2V trained model per training dataset cfg in sqlite
* global config => store experiment prefix => make -e param optional (infer from config name)

## TSA Over Time (/loaded Recordings)

* Plot TSA characteristics in function of loaded recordings/number of system calls

## Advanced Dataloder Features

* load attacks from different scenario
* Combine Scenarios
* Specify Number of Attack Recordings Loaded 
* Load more Data until certain Measure (Entropy, ... (?) ) is reached

## TSA

* model zipf distribution characteristics
* make use of local outlier factor values (mean, variance, ...)
* cluster-wise entropy?

## Robustness

* Check MixedModel implementation
* Add Robust AE implementation as DE

## Experiment Evaluation

* plot total number of false alarms
