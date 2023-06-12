
# June II (12-23)

## Advanced Dataloder Features

* load attacks from different scenario + Combine Scenarios 

## Frequency Embedding

* specify distance metric via parameter ("2max", "max(freq;...)", ...)

## Code Maintainance

* problem: permutation_i = 0 == permutation_i = 1 (nur andere Reihenfolge)
* backup databricks mlflow runs
* test loaded attacks in dataloader

## Experiment Validity

* check command for "search"
* remove duplicates
* download multiple experiments in one csv

## Visualization

* different plot tool? / plot script

--- 

# BACKLOG

## TSA

* LOFAnalyser => make use of local outlier factor values (mean, variance, ...)
* ClusteringAnalyser => cluster-wise entropy?
* entropy values from paper
* model zipf distribution characteristics

## Dataloader

* Load more Data until certain Measure (Entropy, ... (?) ) is reached

## Experiment Evaluation

* plot total number of false alarms

---

# Ideas/Backlog

* Training Set Visualization
* Zipf Law for Training Set?
* dimensions of features vs number of needed training instances?
* date exploration with apache superset, etc.

## Code Maintainance / SWE

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
