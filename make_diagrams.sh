#!/bin/bash

MLFLOW_CACHE=${MLFLOW_CACHE:-results/mlflow_cache}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts}

# f-STIDE

python cli.py eval --config experiments/slurm/f-stide/f-stide-exp-a0.3.yaml experiments/slurm/f-stide/f-stide-exp-a0.5.yaml experiments/slurm/f-stide/f-stide-exp-a0.7.yaml experiments/slurm/f-stide/f-stide-exp-a0.9.yaml experiments/slurm/f-stide/f-stide-exp-a0.95.yaml experiments/slurm/f-stide/f-stide-homographic-a0.5.yaml experiments/slurm/f-stide/f-stide-homographic-a2.yaml experiments/slurm/f-stide/f-stide-homographic-a5.yaml experiments/slurm/f-stide/f-stide-max-scaled.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/f-stide --names exp-0.3 exp-0.5 exp-0.7 exp-0.9 exp-0.95 hom-0.5 hom-2 hom-5 linear

# thread-f-STIDE, tfidf-STIDE

python cli.py eval --config experiments/slurm/thread-f-stide/thread-freq-homographic-a1.yaml experiments/slurm/thread-f-stide/thread-freq-homographic-a2.yaml experiments/slurm/tfidf-stide/tfidf_stide-mean-1.5.yaml experiments/slurm/tfidf-stide/tfidf_stide-mean-1.yaml --cache $MLFLOW_CACHE --artifacts-dir $ARTIFACTS_DIR/thread-based-stide-acc --names thread-f-stide-hom-1 thread-f-stide-hom-2 tfidf-stide-1.5 tfidf-stide-1

# frequency OD

python cli.py eval --config experiments/slurm/preprocessing/frequency-od/rel/stide-0.00005.yaml experiments/slurm/preprocessing/frequency-od/rel/stide-0.00001.yaml experiments/slurm/preprocessing/frequency-od/rel/stide-0.000005.yaml  --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/frequency-od-rel/ --names t=0.005%  t=0.001% t=0.0005%

python cli.py eval --config experiments/slurm/preprocessing/frequency-od/stide-t1.yaml experiments/slurm/preprocessing/frequency-od/stide-t3.yaml experiments/slurm/preprocessing/frequency-od/stide-t8.yaml experiments/slurm/preprocessing/frequency-od/stide-t15.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/frequency-od-abs/ --names t=1 t=3 t=8 t=15

python cli.py eval --config experiments/slurm/preprocessing/frequency-od/stide-t1.yaml experiments/slurm/preprocessing/frequency-od/stide-t3.yaml experiments/slurm/preprocessing/frequency-od/stide-t15.yaml experiments/slurm/preprocessing/frequency-od/rel/stide-0.00001.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/frequency-od-both/ --names abs-t=1 abs-t=3 abs-t=15 rel-t=0.001%

# thread OD

python cli.py eval --config experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n1.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n3.yaml experiments/slurm/baseline/stide.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --cache $MLFLOW_CACHE --artifacts-dir $ARTIFACTS_DIR/thread-od-tfidf --names n=1 n=2 n=3 "baseline (stide)"

python cli.py eval --config experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n1.yaml experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n3.yaml experiments/slurm/baseline/stide.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/thread-od --names n=1 n=2 n=3 "baseline (stide)"

# 
python cli.py eval --config experiments/slurm/preprocessing/thread-od/lof-{binary-jaccard,jds}-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-hellinger-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jds-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-canberra-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-chebyshev-n2.yaml experiments/slurm/baseline/stide.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/thread-od-distances --names binary-hamming jds tfidf+cosine tfidf+jaccard-cosin tfidf+jaccard-hellinger tfidf+jds tfidf+canberra tfidf+chebyshev "baseline (stide)"

# SOM experiments

python cli.py eval --config experiments/slurm/som/sample-0.{5,7,9,95,99}.yaml experiments/slurm/baseline/som.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/som-sample --names b=0.5 b=0.7 b=0.9 b=0.95 b=0.99 "SOM baseline"

python cli.py eval --config experiments/slurm/som/size0.{5,7,9}.yaml experiments/slurm/baseline/som.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/som-size --names s=0.5 s=0.7 s=0.9 "SOM baseline"

python cli.py eval --config experiments/slurm/frequency_append/som-{ngram,thread,ngram-thread}.yaml experiments/slurm/baseline/som.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/som-freq-append --names +ngram +thread +ngram+thread "SOM baseline"

# SCG Accomodation

python cli.py eval --config experiments/slurm/baseline/scg{,-thread-wise-graphs}.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/scg --names "SCG baseline" thread-wise-graphs
