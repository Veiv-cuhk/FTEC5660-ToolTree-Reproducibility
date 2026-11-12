#!/usr/bin/env bash
# Build the FTEC5660 reproducibility report PDF from LaTeX.
# Requirements: TeX Live / MiKTeX with pdflatex (TeX Live 2024 was used).
# Usage (from the repository root):  bash report/build.sh
set -euo pipefail
cd "$(dirname "$0")"

NAME="FTEC5660_Reproducibility_Report_JIANG_Jiawei_1155271183"

pdflatex -interaction=nonstopmode -halt-on-error report.tex > /dev/null
pdflatex -interaction=nonstopmode -halt-on-error report.tex > /dev/null
mv -f report.pdf "${NAME}.pdf"
rm -f report.aux report.log report.out report.toc

echo "Built report/${NAME}.pdf"
