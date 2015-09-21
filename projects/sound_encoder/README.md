
This directory contains an implementation of a sound encoder. A sound wave is
converted into the maximum frequency detected according to FFT, and this
frequency is encoded into an SDR using a ScalarEncoder.

This was done as a small summer internship experiment in 2015. The encoder
implementation itself is in `htmresearch/encoders/sound_encoder`
