#!/bin/bash

set -eux

MLFLOW_CACHE=${MLFLOW_CACHE:-results/mlflow_cache}
ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts}

# baseline

python cli.py eval --config experiments/slurm/baseline/{stide,som,scg}.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/baseline --names STIDE SOM SCG

# combined scenarios

python cli.py eval --config experiments/slurm/combined/baseline/{stide,som,scg}.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/baseline-combined-scenarios --names STIDE SOM SCG
# f-STIDE

python cli.py eval --config experiments/slurm/f-stide/f-stide-homographic-a0.5.yaml experiments/slurm/f-stide/f-stide-homographic-a2.yaml experiments/slurm/f-stide/f-stide-homographic-a5.yaml experiments/slurm/f-stide/f-stide-max-scaled.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/f-stide-hom-linear --names hom-0.5 hom-2 hom-5 linear
python cli.py eval --config experiments/slurm/f-stide/f-stide-exp-a0.3.yaml experiments/slurm/f-stide/f-stide-exp-a0.5.yaml experiments/slurm/f-stide/f-stide-exp-a0.7.yaml experiments/slurm/f-stide/f-stide-exp-a0.9.yaml experiments/slurm/f-stide/f-stide-exp-a0.95.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/f-stide-exp --names exp-0.3 exp-0.5 exp-0.7 exp-0.9 exp-0.95
python cli.py eval --config experiments/slurm/f-stide/f-stide-exp-a0.3.yaml experiments/slurm/f-stide/f-stide-exp-a0.5.yaml experiments/slurm/f-stide/f-stide-exp-a0.7.yaml experiments/slurm/f-stide/f-stide-exp-a0.9.yaml experiments/slurm/f-stide/f-stide-exp-a0.95.yaml experiments/slurm/f-stide/f-stide-homographic-a0.5.yaml experiments/slurm/f-stide/f-stide-homographic-a2.yaml experiments/slurm/f-stide/f-stide-homographic-a5.yaml experiments/slurm/f-stide/f-stide-max-scaled.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/f-stide --names exp-0.3 exp-0.5 exp-0.7 exp-0.9 exp-0.95 hom-0.5 hom-2 hom-5 linear

# thread-f-STIDE, tfidf-STIDE, norm-entropy-stide

python cli.py eval --config experiments/slurm/thread-f-stide/norm_entropy.yaml experiments/slurm/thread-f-stide/thread-freq-homographic-a1.yaml experiments/slurm/thread-f-stide/thread-freq-homographic-a2.yaml experiments/slurm/tfidf-stide/tfidf_stide-mean-1.5.yaml experiments/slurm/tfidf-stide/tfidf_stide-mean-1.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/thread-based-stide-acc --names norm-entropy-stide thread-f-stide-hom-1 thread-f-stide-hom-2 tfidf-stide-1.5 tfidf-stide-1

# frequency OD

python cli.py eval --config experiments/slurm/preprocessing/frequency-od/rel/stide-0.00005.yaml experiments/slurm/preprocessing/frequency-od/rel/stide-0.00001.yaml experiments/slurm/preprocessing/frequency-od/rel/stide-0.000005.yaml  --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/frequency-od-rel/ --names t=0.005%  t=0.001% t=0.0005%

python cli.py eval --config experiments/slurm/preprocessing/frequency-od/stide-t1.yaml experiments/slurm/preprocessing/frequency-od/stide-t3.yaml experiments/slurm/preprocessing/frequency-od/stide-t8.yaml experiments/slurm/preprocessing/frequency-od/stide-t15.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/frequency-od-abs/ --names t=1 t=3 t=8 t=15

python cli.py eval --config experiments/slurm/preprocessing/frequency-od/stide-t1.yaml experiments/slurm/preprocessing/frequency-od/stide-t3.yaml experiments/slurm/preprocessing/frequency-od/stide-t15.yaml experiments/slurm/preprocessing/frequency-od/rel/stide-0.00001.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/frequency-od-both/ --names abs-t=1 abs-t=3 abs-t=15 rel-t=0.001%

# thread OD

python cli.py eval --config experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n1.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n3.yaml experiments/slurm/baseline/stide.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --cache $MLFLOW_CACHE --artifacts-dir $ARTIFACTS_DIR/thread-od-tfidf --names n=1 n=2 n=3 "baseline (stide)"

python cli.py eval --config experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n1.yaml experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/lof-jaccard-cosine-n3.yaml experiments/slurm/baseline/stide.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/thread-od --names n=1 n=2 n=3 "baseline (stide)"

# 
python cli.py eval --config experiments/slurm/preprocessing/thread-od/lof-{binary-jaccard,jds}-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-hellinger-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-jds-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-canberra-n2.yaml experiments/slurm/preprocessing/thread-od/tfidf-lof-chebyshev-n2.yaml experiments/slurm/baseline/stide.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/thread-od-distances --names binary-hamming "jsd" tfidf+cosine tfidf+jaccard-cosin tfidf+jaccard-hellinger tfidf+jsd tfidf+canberra tfidf+chebyshev "baseline (stide)"

# SOM experiments

python cli.py eval --config experiments/slurm/som/sample-0.{5,7,9,95,99}.yaml experiments/slurm/baseline/som.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/som-sample --names b=0.5 b=0.7 b=0.9 b=0.95 b=0.99 "SOM baseline"

python cli.py eval --config experiments/slurm/som/size0.{5,7,9}.yaml experiments/slurm/baseline/som.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/som-size --names s=0.5 s=0.7 s=0.9 "SOM baseline"

python cli.py eval --config experiments/slurm/frequency_append/som-{ngram,thread,ngram-thread}.yaml experiments/slurm/baseline/som.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/som-freq-append --names +ngram +thread +ngram+thread "SOM baseline"

# SOM combination
python cli.py eval --config experiments/slurm/som/sample-{0.95,0.99}.yaml experiments/slurm/som/sample-{0.95,0.99}+thread-size0.7.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/som-combination --names "Only Sampling (b=0.95)" "Only Sampling (b=0.99)" "Combination I" "Combination II"

# SCG Accomodation
python cli.py eval --config experiments/slurm/baseline/scg{,-thread-wise-graphs}.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/scg --names "SCG baseline" thread-wise-graphs

# Combination THread-OD + thread-f-stide

python cli.py eval --config experiments/slurm/preprocessing/thread-od/tfidf-lof-jaccard-cosine.yaml experiments/slurm/thread-f-stide/thread-freq-homographic-a2.yaml experiments/slurm/combination/stide/lof-cosine+thread-freq-homographic-a2.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/combination-stide-thread-od --names "Only Thread OD" "Only thread-f-stide" "Combination"

# Combination THread-OD + SCG

python cli.py eval --config experiments/slurm/baseline/scg{,-thread-wise-graphs}.yaml experiments/slurm/combination/scg/lof-cosine+scg-baseline.yaml  experiments/slurm/combination/scg/lof-cosine+thread-wise.yaml --cache $MLFLOW_CACHE $ADDITIONAL_OPTIONS --artifacts-dir $ARTIFACTS_DIR/combination-scg-thread-od --names "SCG baseline" "Only SCG with thread-wise graphs" "Thread-OD + SCG baseline" "Thread OD + SCG with thread-wise graphs"

for f in $(find $ARTIFACTS_DIR -type f -name '*.svg'); do
  rsvg-convert -f pdf -o $f.pdf $f
  pdfcrop --margin 1 $f.pdf $f.pdf
done