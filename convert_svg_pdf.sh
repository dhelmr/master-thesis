ARTIFACTS_DIR=${ARTIFACTS_DIR:-results/test-artifacts/rsq2-2}

for f in $(find $ARTIFACTS_DIR -type f -name '*.svg'); do
  rsvg-convert -f pdf -o $f.pdf $f
  pdfcrop --margin 1 $f.pdf $f.pdf
done