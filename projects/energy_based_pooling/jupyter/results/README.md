
# How to interpret the images

## Reconstruction

Each row shows an excerpt of a binary vector. The first 50 entries in a row encode the x-coordinate, and the last 50 bits encode the y-coordinate of a point in [0,1]x[0,1].
The last 50 entries in each row are reapeted 2 more times to form the full binary input vector (i.e. the actual length of the vectors is 100 + 2*50 = 200).

 - Right: 25 test input vectors.
 - Left: Reconstruction of the corresponding input on the right.

<p align="center"><img src="../../media/reconstruction_example.png"></p>

## Scatter

Each point corresponds to a learned feature (i.e. a row in the weight matrix W encoding the visible-to-hidden connections).

- x-coordinate: sum of the weights of connections to the first 50 input units
- y-coordinate: sum of the weights of connections to the last 150 input units

<p align="center"><img src="../../media/scatter_example.png"></p>


## Learned features

Each column corresponds to a learned feature (i.e. a row in the weight matrix W encoding the visible-to-hidden connections).

<p align="center"><img src="../../media/features_example.png"></p>



## ...

