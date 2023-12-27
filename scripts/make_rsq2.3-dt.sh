set -eux
ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts/rsq2-3}
mkdir -p "$ARTIFACTS_DIR"

function make_tree {
FEATURES=$(pd -i results/rsq2-2/$1-best-concat.csv --sort "mean precision" -q "max_depth == 2"  --descending --head 1 --unique features | sed 's/;/ /g')
# DEPTH=$(pd -i results/rsq2-2/$1-best-concat.csv --sort "mean precision" -q "max_depth == 2" --descending --head 1 --unique max_depth)

python cli.py tsa-ruleminer -i results/rsq2-2/$1.csv -f ${FEATURES} -p DecisionTree --max-depth 2 -o $ARTIFACTS_DIR/$1.svg $2
}

make_tree perf-ngrams-augmented ""
make_tree perf-threads-n3-augmented ""
make_tree dd-threads-n3-augmented-jsd "--target jensen_shannon_divergence --threshold 0.0039 --reverse-classes"
make_tree dd-threads-n3-augmented-rutn "--target ratio_unseen_test_ngrams --threshold 0.0078 --reverse-classes"

make_tree dd-ngrams-augmented-jsd "--target jensen_shannon_divergence --threshold 0.0039 --reverse-classes"
make_tree dd-ngrams-augmented-rutn "--target ratio_unseen_test_ngrams --threshold 0.0078 --reverse-classes"