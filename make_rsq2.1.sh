set -eux

ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts/rsq2-1}

mkdir -p "$ARTIFACTS_DIR"

pd -i results/rsq2-1/perf-combined-features.corr-dr.csv --tail 30 --to latex > "$ARTIFACTS_DIR/corr-dr.tex"
pd -i results/rsq2-1/perf-combined-features.corr-f1.csv --tail 30 --to latex > "$ARTIFACTS_DIR/corr-f1.tex"
pd -i results/rsq2-1/perf-combined-features.corr-pr.csv --tail 30 --to latex > "$ARTIFACTS_DIR/corr-pr.tex"
pd -i results/rsq2-1/dd-combined-features.corr-rutn-unique.csv --tail 30 --to latex > "$ARTIFACTS_DIR/corr-rutn-unique.tex"
pd -i results/rsq2-1/dd-combined-features.corr-rutn.csv --tail 30 --to latex > "$ARTIFACTS_DIR/corr-rutn.tex"
pd -i results/rsq2-1/dd-combined-features.corr-js-distance.csv --tail 30 --to latex > "$ARTIFACTS_DIR/corr-js-distance.tex"


pd -i results/rsq2-1/dd-combined-features.corr-js-divergence.csv --tail 30 --to latex > "$ARTIFACTS_DIR/corr-js-divergence.tex"
