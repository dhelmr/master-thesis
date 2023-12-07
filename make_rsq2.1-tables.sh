set -eux

ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts/rsq2-1}

mkdir -p "$ARTIFACTS_DIR"

pd -i results/rsq2-1/perf.corr-dr.csv --tail 25 --to latex > "$ARTIFACTS_DIR/corr-dr.tex"
pd -i results/rsq2-1/perf.corr-f1.csv --tail 25 --to latex > "$ARTIFACTS_DIR/corr-f1.tex"
pd -i results/rsq2-1/perf.corr-pr.csv --tail 25 --to latex > "$ARTIFACTS_DIR/corr-pr.tex"
pd -i results/rsq2-1/dd.corr-ruutn.csv --tail 25 --to latex > "$ARTIFACTS_DIR/corr-rutn-unique.tex"
pd -i results/rsq2-1/dd.corr-rutn.csv --tail 25 --to latex > "$ARTIFACTS_DIR/corr-rutn.tex"
pd -i results/rsq2-1/dd.corr-js-distance.csv --tail 25 --to latex > "$ARTIFACTS_DIR/corr-js-distance.tex"
pd -i results/rsq2-1/dd.corr-js-divergence.csv --tail 25 --to latex > "$ARTIFACTS_DIR/corr-js-divergence.tex"
