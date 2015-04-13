The MNIST data set is publicly available database of
handwritten digits containing 60,000 training samples
and 10,000 testing samples. The MNIST data has
been packaged and published by Dr. Yann LeCun of
the Courant Institute.

Instructions:

1. Download the following four MNIST archive files from
   Yann LeCun's website:

```
   export LECUNSITE=http://yann.lecun.com/exdb/mnist

   On MacOS:
     curl $LECUNSITE/train-images-idx3-ubyte.gz -o mnist_extraction_source/train-images-idx3-ubyte.gz
     curl $LECUNSITE/train-labels-idx1-ubyte.gz -o mnist_extraction_source/train-labels-idx1-ubyte.gz
     curl $LECUNSITE/t10k-images-idx3-ubyte.gz -o  mnist_extraction_source/t10k-images-idx3-ubyte.gz
     curl $LECUNSITE/t10k-labels-idx1-ubyte.gz -o  mnist_extraction_source/t10k-labels-idx1-ubyte.gz

   On Linux:
     wget $LECUNSITE/train-images-idx3-ubyte.gz
     wget $LECUNSITE/train-labels-idx1-ubyte.gz
     wget $LECUNSITE/t10k-images-idx3-ubyte.gz
     wget $LECUNSITE/t10k-labels-idx1-ubyte.gz

2. Decompress the tarred archives files:

   gunzip train-images-idx3-ubyte.gz
   gunzip train-labels-idx1-ubyte.gz
   gunzip t10k-images-idx3-ubyte.gz
   gunzip t10k-labels-idx1-ubyte.gz

3. Build and invoke the "extractMNIST" binary to
   extract the pixel values and associated category labels
   from the monolithic archive into individual text files
   (one file per training or testing sample.)

```

The result will be a set of 70,000 text-format images
organized by category (0 through 9) and partitioned into
60,000 training images and 10,000 testing images.

Acknowledgements
+=+=+=+=+=+=+=+=+
Numenta wishes to express its gratitude to Dr. Yann LeCun
of the Courant Institute of Mathematical Sciences,
New York, NY, for publishing the MNIST data set.
Details regarding MNIST and related publications are
available at the following URL: http://yann.lecun.com/exdb/mnist
