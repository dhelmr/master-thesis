set -eux
ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts/rsq2-2}
mkdir -p "$ARTIFACTS_DIR"

CLASS_1="class 1 (suitable)"
CLASS_0="class 0 (not suitable)"

python cli.py tsa-stats -i results/rsq2-2/dd-ngrams.csv --target jensen_shannon_divergence --threshold 0.0039 -o results/rsq2-2/stats-jsd.csv
python cli.py tsa-stats -i results/rsq2-2/dd-ngrams.csv --target ratio_unseen_test_ngrams --threshold 0.0078 -o results/rsq2-2/stats-rutn.csv
python cli.py tsa-stats -i results/rsq2-2/perf-ngrams.csv --target f1_cfa --threshold 0.8 -o results/rsq2-2/stats-f1.csv

pd -i results/rsq2-2/stats-f1.csv --rename unique_timepoints::measurements f1_cfa-mean::f1-mean "f1_cfa>0.8::$CLASS_1" "f1_cfa<=0.8::$CLASS_0" --only scenario f1-mean "$CLASS_1" "$CLASS_0"  --to latex -o $ARTIFACTS_DIR/stats-f1.tex
pd -i results/rsq2-2/stats-rutn.csv --rename unique_timepoints::measurements "ratio_unseen_test_ngrams>0.0078::$CLASS_0" "ratio_unseen_test_ngrams<=0.0078::$CLASS_1" --only scenario ratio_unseen_test_ngrams-mean "$CLASS_1" "$CLASS_0"  --to latex -o $ARTIFACTS_DIR/stats-rutn.tex
pd -i results/rsq2-2/stats-jsd.csv --rename unique_timepoints::measurements "jensen_shannon_divergence>0.0039::$CLASS_0" "jensen_shannon_divergence<=0.0039::$CLASS_1" --only scenario jensen_shannon_divergence-mean "$CLASS_1" "$CLASS_0"  --to latex -o $ARTIFACTS_DIR/stats-jsd.tex