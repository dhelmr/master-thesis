set -eux
ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts/rsq2-2}
mkdir -p "$ARTIFACTS_DIR"

python cli.py tsa-cv -i results/rsq2-2/perf-ngrams.csv -p BaselineRandom BaselineAlways1 BaselineAlways0  -o results/rsq2-2/baseline-f1-ngrams.csv
python cli.py tsa-cv -i results/rsq2-2/dd-ngrams.csv -p BaselineRandom BaselineAlways1 BaselineAlways0 -o results/rsq2-2/baseline-rutn-ngrams.csv -s ratio_unseen_test_ngrams --target ratio_unseen_test_ngrams --threshold 0.0078
python cli.py tsa-cv -i results/rsq2-2/dd-ngrams.csv -p BaselineRandom BaselineAlways1 BaselineAlways0 -o results/rsq2-2/baseline-jsd-ngrams.csv -s jensen_shannon_divergence --target jensen_shannon_divergence --threshold 0.0039

pd -i results/rsq2-2/baseline-f1-ngrams.csv --only predictor "mean.precision" "mean.tnr" "mean.f1_score" --to latex -o "$ARTIFACTS_DIR/baseline-f1-ngram.tex"
pd -i results/rsq2-2/baseline-jsd-ngrams.csv --only predictor "mean.precision" "mean.tnr" "mean.f1_score"   --to latex -o "$ARTIFACTS_DIR/baseline-jsd-ngram.tex"
pd -i results/rsq2-2/baseline-rutn-ngrams.csv --only predictor "mean.precision" "mean.tnr" "mean.f1_score"  --to latex -o "$ARTIFACTS_DIR/baseline-rutn-ngram.tex"