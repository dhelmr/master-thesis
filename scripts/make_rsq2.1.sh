set -eux
# add suffix to thread-ngram features in order to distinguish them
# "--skip-features unique_ngrams/total unique_ngrams" is set because these statistics are duplicated in the datasets
IGNORE_ZIPF_FEATURES="ngram_dists-zipf_a_mean ngram_dists-zipf_a_var ngram_dists-zipf_a_min ngram_dists-zipf_a_max ngram_dists-zipf_a_median ngram_dists-zipf_a_iod ngram_dists-zipf_loc_mean ngram_dists-zipf_loc_var ngram_dists-zipf_loc_min ngram_dists-zipf_loc_max ngram_dists-zipf_loc_median ngram_dists-zipf_loc_iod thread_dists-zipf_a_mean thread_dists-zipf_a_var thread_dists-zipf_a_min thread_dists-zipf_a_max thread_dists-zipf_a_median thread_dists-zipf_a_iod thread_dists-zipf_loc_mean thread_dists-zipf_loc_var thread_dists-zipf_loc_min thread_dists-zipf_loc_max thread_dists-zipf_loc_median thread_dists-zipf_loc_iod"
python cli.py tsa-add-suffix -i results/thread_matrix-n2.csv --suffix '@n2' -o results/rsq2-1/thread_matrix-n2-suffix.csv --skip-features unique_ngrams/total unique_ngrams total $IGNORE_ZIPF_FEATURES
python cli.py tsa-add-suffix -i results/thread_matrix-n3.csv --suffix '@n3' -o results/rsq2-1/thread_matrix-n3-suffix.csv --skip-features unique_ngrams/total unique_ngrams total $IGNORE_ZIPF_FEATURES

python cli.py tsa-ngram-auc -i results/analysis-all-ngrams.csv -o results/rsq2-1/analysis-all-ngrams-auc.csv --keep-ngram-size 1 2 5 10 15 25

python cli.py tsa-concat -i results/rsq2-1/analysis-all-ngrams-auc.csv results/rsq2-1/thread_matrix-n2-suffix.csv results/rsq2-1/thread_matrix-n3-suffix.csv results/data-drift-no-attacks.csv -o results/rsq2-1/dd.csv --common syscalls --skip unique_ngrams/total unique_ngrams total $IGNORE_ZIPF_FEATURES

python cli.py tsa-combine --statistics-csv results/rsq2-1/dd.csv -e $EXPERIMENT_PREFIX/analysis-stide-max_syscalls.search.yaml -o results/rsq2-1/perf.csv

python cli.py tsa-augment -i results/rsq2-1/perf.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_unique_test_ngrams ratio_unseen_test_ngrams -a FeatureTransform -o results/rsq2-1/perf-augmented.csv
#python cli.py tsa-augment -i results/rsq2-1/perf-transformed.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_unique_test_ngrams ratio_unseen_test_ngrams -a FeatureCombine -o results/rsq2-1/perf-transformed+combined.csv
#python cli.py tsa-augment -i results/rsq2-1/perf.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_unique_test_ngrams ratio_unseen_test_ngrams -a FeatureCombine -o results/rsq2-1/perf-combined.csv
#python cli.py tsa-concat -i results/rsq2-1/perf-combined.csv results/rsq2-1/perf-transformed.csv -o results/rsq2-1/perf-augmented.csv --common syscalls

python cli.py tsa-augment -i results/rsq2-1/dd.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_unique_test_ngrams ratio_unseen_test_ngrams -a FeatureTransform -o results/rsq2-1/dd-augmented.csv
#python cli.py tsa-augment -i results/rsq2-1/dd.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_unique_test_ngrams ratio_unseen_test_ngrams -a FeatureCombine -o results/rsq2-1/dd-combined.csv
#python cli.py tsa-concat -i results/rsq2-1/dd-combined.csv results/rsq2-1/dd-transformed.csv -o results/rsq2-1/dd-augmented.csv --common syscalls
#python cli.py tsa-augment -i results/rsq2-1/dd-transformed.csv --skip-features jensen_shannon_divergence jensen_shannon_distance ratio_unseen_unique_test_ngrams ratio_unseen_test_ngrams -a FeatureCombine -o results/rsq2-1/dd-transformed+combined.csv
