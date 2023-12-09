set -eux
ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts/rsq2-2}
RSQ_DIR="results/rsq2-2"
mkdir -p "$ARTIFACTS_DIR"

function make-table { # $1 = basename file prefix (e.g. perf-ngrams-augmented) $2=best baseline f1 score
  python cli.py tsa-eval-fs -f results/rsq2-2/fs/$1* --input results/rsq2-2/$1.csv -o $RSQ_DIR/$1-best.csv -q "`gain.precision` > 0.03 and `mean.f1_score` >= $2" --add-max-depth-column
  pd -i $RSQ_DIR/$1-best.csv --sort "mean.precision" --tail 5 --only max_depth features mean.precision mean.tnr mean.f1_score mean.mcc gain.precision -q "max_depth == 2" -o $RSQ_DIR/$1-best-depth=2.csv
    pd -i $RSQ_DIR/$1-best.csv --sort "mean.precision" --tail 5 --only max_depth features mean.precision mean.tnr mean.f1_score mean.mcc gain.precision -q "max_depth == 3" -o $RSQ_DIR/$1-best-depth=3.csv
    pd -i $RSQ_DIR/$1-best.csv --sort "mean.precision" --tail 5 --only max_depth features mean.precision mean.tnr mean.f1_score mean.mcc gain.precision -q "max_depth == 5" -o $RSQ_DIR/$1-best-depth=5.csv
    pd -i $RSQ_DIR/$1-best.csv --sort "mean.precision" --tail 5 --only max_depth features mean.precision mean.tnr mean.f1_score mean.mcc gain.precision -q "max_depth == 10" -o $RSQ_DIR/$1-best-depth=10.csv
    pd -i $RSQ_DIR/$1-best.csv --sort "mean.precision" --tail 5 --only max_depth features mean.precision mean.tnr mean.f1_score mean.mcc gain.precision -q "max_depth == 20" -o $RSQ_DIR/$1-best-depth=20.csv
  python concat-csv.py -i $RSQ_DIR/$1-best-depth={2,3,5,10,20}.csv -o $RSQ_DIR/$1-best-concat.csv
  pd -i $RSQ_DIR/$1-best-concat.csv --to latex -o $ARTIFACTS_DIR/rsq2-2-$1-best.tex
}

make-table perf-ngrams-augmented
make-table perf-threads-n3-augmented
make-table dd-ngrams-augmented-jsd
make-table dd-threads-n3-augmented-jsd
make-table dd-ngrams-augmented-rutn
make-table dd-threads-n3-augmented-rutn
