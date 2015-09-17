# This variable controls where we download the MNIST files from
LECUNSITE=http://yann.lecun.com/exdb/mnist

# Get into the extraction directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
pushd $DIR/mnist_extraction_source

# Download the image data
curl $LECUNSITE/train-images-idx3-ubyte.gz -o train-images-idx3-ubyte.gz
curl $LECUNSITE/train-labels-idx1-ubyte.gz -o train-labels-idx1-ubyte.gz
curl $LECUNSITE/t10k-images-idx3-ubyte.gz -o t10k-images-idx3-ubyte.gz
curl $LECUNSITE/t10k-labels-idx1-ubyte.gz -o t10k-labels-idx1-ubyte.gz

# Extract the archives
gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

# Build the C++ script for extracting features from the images
./build.sh

# Extract the features from the images
./extractMNIST

# Clean up
popd
