#!/bin/bash

set -e -x

# Columns+ chart processing

# Labels
# To see font list: `convert -list font`
echo "a" | magick convert -background white -fill black -pointsize 36 -font "TimesNewRoman" -density 300 text:- -trim +repage -bordercolor white -border 3 a.pdf
echo "b" | magick convert -background white -fill black -pointsize 36 -font "TimesNewRoman" -density 300 text:- -trim +repage -bordercolor white -border 3 b.pdf

# Figure 5
magick montage -density 300 a.pdf narrowing_singleTrials_gaussian.pdf b.pdf narrowing_aggregated_gaussian.pdf -tile 1x4 -geometry +1+4 figure5.pdf

# Old commands from previous paper as examples

## Figure 1
#magick convert -density 300 -alpha off -compress zip fig1.pdf fig1.tiff
#
## Figure 2
#magick convert -density 300 -alpha off -compress zip fig2.pdf fig2.tiff
#
## Figure 5
#magick montage -density 300 a.pdf fig5a.pdf b.pdf fig5b.pdf -tile 2x2 -geometry +2+2 fig5.pdf
#magick convert -density 300 -alpha off -compress zip fig5.pdf fig5.tiff
#
## Figure 6
#magick montage -density 300 a.pdf fig6a.pdf b.pdf fig6b.pdf c.pdf fig6c.pdf -tile 2x3 -geometry +2+2 fig6.pdf
#magick convert -density 300 -alpha off -compress zip fig6.pdf fig6.tiff
#
## Figure 7
#magick montage -density 300 fig7a.pdf fig7b.pdf fig7f.pdf fig7c.pdf fig7e.pdf fig7d.pdf -tile 2x3 -geometry +2+2 fig7.pdf
#magick convert -density 300 -alpha off -compress zip fig7.pdf fig7.tiff
#
## Figure 8
#magick convert -density 300 -alpha off -compress zip fig8.pdf fig8.tiff
