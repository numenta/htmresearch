#!/bin/bash

set -e -x

# Columns+ chart processing

# Labels
# To see font list: `convert -list font`
echo "A" | magick convert -background white -fill black -pointsize 24 -font "Arial" -density 400 text:- -trim +repage -bordercolor white -border 3 a.tiff
echo "B" | magick convert -background white -fill black -pointsize 24 -font "Arial" -density 400 text:- -trim +repage -bordercolor white -border 3 b.tiff

# Figure 5
magick montage -density 400 a.tiff charts/narrowing_singleTrials_gaussian.pdf b.tiff charts/narrowing_aggregated_gaussian.pdf -compress zip -tile 4x1 -geometry +4+1 -gravity north charts/figure5.tiff

# Figure 6
magick convert -density 400 charts/comparisonToIdeal_gaussian.pdf -compress zip charts/figure6.tiff

# Figure 7
magick montage -density 400 a.tiff charts/capacityTrends_gaussian.pdf b.tiff charts/capacityHeatmap_gaussian.pdf -compress zip -tile 2x2 -geometry +2+2 charts/figure7.tiff

# Figure 8
magick convert -density 400 charts/summary_gaussian.pdf -compress zip charts/figure8.tiff

# Figure S1
magick montage -density 400 a.tiff charts/featureDistributions_gaussian_1.pdf b.tiff charts/featureDistributions_gaussian_2.pdf -compress zip -tile 4x1 -geometry +4+1 -gravity north charts/figureS1.tiff
