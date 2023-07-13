Robustness studies and training set analysis for HIDS
# July I (03-12)





---

# JULY II (12-21) 

## Various

* T-Stide: relative frequency


## Experiment Building

* Decision Engine as Outlier Detector 
* Multiply/ADD FrequencySTIDE to SOM Score

## Code Maintainance

* backup databricks mlflow runs
* test loaded attacks in dataloader

## Similar Normal Behaviour (2019)

* CVE-2014-0160 + CWE-307 (Hearbleed, Brute Force Login) => Apache with vulnerable OpenSSL version; Selenium
* CWE-89 + CWE-434 (SQL Injection, PHP File Upload)
* CVE-2014-3120 + CVE-2015-1427 (Arbitrary Code Execution)
* CVE-2016-6515 + CVE-2015-5602 (Local Privilege Escalation)
* Zip Slip + SWE-434 (Zip Slip, PS)


--- 

# BACKLOG

* f-stide alpha anhand von training set eigenschaften bestimmen

## TSA

* LOFAnalyser => make use of local outlier factor values (mean, variance, ...)
* ClusteringAnalyser => cluster-wise entropy?
* entropy values from paper
* model zipf distribution characteristics

## Dataloader

* Load more Data until certain Measure (Entropy, ... (?) ) is reached

## Experiment Evaluation

* plot total number of false alarms

# important
`
* TSA auf kompletten Scenario (t+v+t) ausführen => erreichen unique n-grams plateau?

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
