set -eux

SKIP_FEATURES="ratio_unseen_test_ngrams jensen_shannon_distance jensen_shannon_divergence ratio_unseen_unique_test_ngrams"
mkdir -p results/rsq2-2/fs/

function run { # args: [feature file] [max depth of decision tree] [total number of features to select] [additional args]
SLURM_CLUSTER=clara-long SLURM_MEM_GB=30 SLURM_HOURS=60 slurm/run.sh tsa-fs -i results/rsq2-2/$1.csv -o results/rsq2-2/fs/$1-depth=$2rounds=$3.csv -p DecisionTree --skip-features $SKIP_FEATURES --max-depth $2 --minmax-scaling --total $3 $4
}

function iterdepth {
  run $1 2 4 "$2"
  run $1 3 4 "$2"
  run $1 5 4 "$2"
  run $1 10 4 "$2"
  run $1 20 4 "$2"
}

ln -sf dd-ngrams-augmented.csv results/rsq2-2/dd-ngrams-augmented-jsd.csv
ln -sf dd-ngrams-augmented.csv results/rsq2-2/dd-ngrams-augmented-rutn.csv
ln -sf dd-threads-n3-augmented.csv results/rsq2-2/dd-threads-n3-augmented-jsd.csv
ln -sf dd-threads-n3-augmented.csv results/rsq2-2/dd-threads-n3-augmented-rutn.csv

iterdepth dd-ngrams-augmented-jsd "--target jensen_shannon_divergence --threshold 0.0039 --reverse-classes"
iterdepth dd-ngrams-augmented-rutn "--target ratio_unseen_test_ngrams --threshold 0.0078 --reverse-classes"
iterdepth dd-threads-n3-augmented-jsd "--target jensen_shannon_divergence --threshold 0.0039 --reverse-classes"
iterdepth dd-threads-n3-augmented-rutn "--target ratio_unseen_test_ngrams --threshold 0.0078 --reverse-classes"

iterdepth perf-ngrams-augmented "--target f1_cfa --threshold 0.8"
iterdepth perf-threads-n3-augmented  "--target f1_cfa --threshold 0.8"