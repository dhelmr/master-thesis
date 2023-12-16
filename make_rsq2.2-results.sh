#!/bin/bash
# this script expects the feature selection (fs) results that are generated with make_rsq2.2-slurmjobs.sh
# they must be copied from the slurm cluster to the directory results/rsq2-2/fs/

set -eux
ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts/rsq2-2}
RSQ_DIR="results/rsq2-2"
mkdir -p "$ARTIFACTS_DIR"


function make-table { # $1 = basename file prefix (e.g. perf-ngrams-augmented) $2=best baseline f1 score
  QUERY=$(printf '`gain.precision` > 0.03 and `mean.f1_score` >= %s' $2)
  TMP_DIR=$(mktemp -d)

  python cli.py tsa-eval-fs -f results/rsq2-2/fs/$1* --input results/rsq2-2/$1.csv -o $TMP_DIR --add-max-depth-column -q "$QUERY"

  pd -i $TMP_DIR/results.csv --sort "mean.precision" --descending --head 5 --only max_depth features "mean precision" "mean tnr" "mean f1-score" "precision gain" --rename "mean.precision::mean precision" "mean.tnr::mean tnr" "mean.f1_score::mean f1-score" "gain.precision::precision gain" -q "max_depth == 2" -o $RSQ_DIR/$1-best-depth=2.csv
  pd -i $TMP_DIR/results.csv --sort "mean.precision" --descending --head 5 --only max_depth features "mean precision" "mean tnr" "mean f1-score" "precision gain"  --rename "mean.precision::mean precision" "mean.tnr::mean tnr" "mean.f1_score::mean f1-score" "gain.precision::precision gain" -q "max_depth == 3" -o $RSQ_DIR/$1-best-depth=3.csv
  pd -i $TMP_DIR/results.csv --sort "mean.precision" --descending --head 5 --only max_depth features "mean precision" "mean tnr" "mean f1-score" "precision gain"  --rename "mean.precision::mean precision" "mean.tnr::mean tnr" "mean.f1_score::mean f1-score" "gain.precision::precision gain" -q "max_depth == 5" -o $RSQ_DIR/$1-best-depth=5.csv
  pd -i $TMP_DIR/results.csv --sort "mean.precision" --descending --head 5 --only max_depth features "mean precision" "mean tnr" "mean f1-score" "precision gain"  --rename "mean.precision::mean precision" "mean.tnr::mean tnr" "mean.f1_score::mean f1-score" "gain.precision::precision gain" -q "max_depth == 10" -o $RSQ_DIR/$1-best-depth=10.csv
  pd -i $TMP_DIR/results.csv --sort "mean.precision" --descending --head 5 --only max_depth features "mean precision" "mean tnr" "mean f1-score" "precision gain"  --rename "mean.precision::mean precision" "mean.tnr::mean tnr" "mean.f1_score::mean f1-score" "gain.precision::precision gain" -q "max_depth == 20" -o $RSQ_DIR/$1-best-depth=20.csv

  python concat-csv.py -i $RSQ_DIR/$1-best-depth={2,3,5,10,20}.csv -o $RSQ_DIR/$1-best-concat.csv

  pd -i $RSQ_DIR/$1-best-concat.csv --sort "mean precision" --descending --only max_depth features "mean precision" "mean tnr" --to latex -o $ARTIFACTS_DIR/rsq2-2-$1-best.tex
}

make-table perf-ngrams-augmented 0.44
make-table perf-threads-n3-augmented 0.44
#make-table dd-ngrams-augmented-jsd 0.35
make-table dd-threads-n3-augmented-jsd 0.35
#make-table dd-ngrams-augmented-rutn 0.25
make-table dd-threads-n3-augmented-rutn 0.25
