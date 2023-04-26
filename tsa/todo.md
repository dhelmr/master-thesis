* refactor creation of building blocks (more generic)
* store building blocks config as mlflow parameters (generic)
* experiments with t-stide, LOF, Mixed Model

# MS April

## Evaluate robustness of baseline approaches (Stide, SCG, SOM)

* Add SCG, SOM to experiments
* Recording Order permutation (deter. + non-repetitive)
* Run experiments on clara
* Calculate robustness measure for experiment (mean f1-score; area-under-f1-score; ...)
* draw graphs from experiment (f1-scores x number of attacks)

## Unsupervised Evaluation of Preprocessing Approaches (Mixed Model, LOF, T-Stide)

* Implement evaluation for unsupervised preprocessors 
* Check LOF Performance (faster)
* Implement T-Stide
* Check Mixed Model implementation
* Run experiments on clara

## Code refactoring

* Refactor attack-mixin file
* 

