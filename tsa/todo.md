
# July I (03-12)

## Various

* W2V Cache for SOM Experiments
* T-Stide: relative frequency

## Code Maintainance

* backup databricks mlflow runs
* test loaded attacks in dataloader

## Experiment Building

* Decision Engine as Outlier Detector 
* Multiply/ADD FrequencySTIDE to SOM Score

## Visualization / Evaluation

* calculate quantitative robustness measure(s) with different attack probabilitiy profiles

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
