set -eux

SKIP_FEATURES="ratio_unseen_test_ngrams jensen_shannon_distance jensen_shannon_divergence ratio_unseen_unique_test_ngrams"
mkdir -p results/rsq2-2/fs/

function run { # args: [feature file] [max depth of decision tree] [total number of features to select] [additional args]
SLURM_MEM_GB=30 SLURM_HOURS=48 slurm/run.sh tsa-fs -i results/rsq2-2/$1.csv -o results/rsq2-2/fs/$1-augmented.csv -p DecisionTree --skip-features $SKIP_FEATURES --max-depth $2 --total $3 $4
}

function iterdepth {
  run $1 2 5 "$2"
  run $1 3 5 "$2"
  run $1 5 5 "$2"
  run $1 10 5 "$2"
  run $1 20 5 "$2"
}

iterdepth dd-ngrams-augmented "--target jensen_shannon_divergence --threshold 0.003 --reverse-classes"
iterdepth dd-ngrams-augmented "--target ratio_unseen_test_ngrams --threshold 0.005 --reverse-classes"
iterdepth dd-threads-n3-augmented "--target jensen_shannon_divergence --threshold 0.003 --reverse-classes"
iterdepth dd-threads-n3-augmented "--target ratio_unseen_test_ngrams --threshold 0.005 --reverse-classes"

iterdepth perf-ngrams-augmented "--target f1_cfa --threshold 0.8"
iterdepth perf-threads-n3-augmented  "--target f1_cfa --threshold 0.8"