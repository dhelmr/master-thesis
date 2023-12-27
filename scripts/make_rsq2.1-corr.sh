COMMON_ARGS="--scenario-mean"

python cli.py tsa-correlate -i results/rsq2-1/dd-augmented.csv -o results/rsq2-1/dd.corr-js-divergence.csv --target jensen_shannon_divergence --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_test_ngrams ratio_unseen_unique_test_ngrams $COMMON_ARGS

python cli.py tsa-correlate -i results/rsq2-1/dd-augmented.csv -o results/rsq2-1/dd.corr-js-distance.csv --target jensen_shannon_distance --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_test_ngrams ratio_unseen_unique_test_ngrams $COMMON_ARGS

python cli.py tsa-correlate -i results/rsq2-1/dd-augmented.csv -o results/rsq2-1/dd.corr-ruutn.csv --target ratio_unseen_unique_test_ngrams --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_test_ngrams ratio_unseen_unique_test_ngrams $COMMON_ARGS
python cli.py tsa-correlate -i results/rsq2-1/dd-augmented.csv -o results/rsq2-1/dd.corr-rutn.csv --target ratio_unseen_test_ngrams --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_test_ngrams ratio_unseen_unique_test_ngrams $COMMON_ARGS

python cli.py tsa-correlate -i results/rsq2-1/perf-augmented.csv -o results/rsq2-1/perf.corr-f1.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_test_ngrams ratio_unseen_unique_test_ngrams --target f1_cfa $COMMON_ARGS
python cli.py tsa-correlate -i results/rsq2-1/perf-augmented.csv -o results/rsq2-1/perf.corr-pr.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_test_ngrams ratio_unseen_unique_test_ngrams --target precision_with_cfa $COMMON_ARGS
python cli.py tsa-correlate -i results/rsq2-1/perf-augmented.csv -o results/rsq2-1/perf.corr-dr.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_test_ngrams ratio_unseen_unique_test_ngrams --target detection_rate $COMMON_ARGS
