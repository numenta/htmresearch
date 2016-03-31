/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014-2015, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU Affero Public License for more details.
 *
 * You should have received a copy of the GNU Affero Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------------
 */

/* ---------------------------------------------------------------------
 * This file runs the MNIST dataset using a simple model composed of a
 * set of dendrites. Each dendrite randomly samples pixels from one image.
 *
 * Key parameters:
 *     Binarization threshold
 *     Number of synapses per dendrite
 *     Threshold
 *     min threshold
 * ----------------------------------------------------------------------
 */

#include <assert.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <vector>
#include <utility>

#include <nupic/math/Math.hpp>
#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SparseMatrix01.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Random.hpp>

using namespace std;
using namespace nupic;

// Populate choices with a random selection of nChoices elements from
// the given row in the given sparse matrix. Throws exception when
// nPopulation < nChoices. This function is used to randomly sample synapses
// from an image.
//
template <typename ChoicesIter>  void sample(SparseMatrix01<UInt, Int> *sm, UInt32 row,
            ChoicesIter choices, UInt32 nChoices, Random &r)
{
  // Get our set of non-zero indices as the population we will sample from
  UInt32 nPopulation = sm->nNonZerosRow(row);
  vector<UInt> population;
  population.resize(nPopulation);
  sm->getRowSparse(row, population.begin());

  // Print non-zeros
  //  cout << "row " << row << "\n";
  //  for (int i=0; i <= nPopulation; i++) cout << population[i] << " ";
  //  cout << "\n";

  if (nChoices == 0)
  {
    return;
  }
  if (nChoices > nPopulation)
  {
    NTA_THROW << "Error: population size " << nPopulation
              << " cannot be greater than number of choices "
              << nChoices << "\n";
  }

  UInt32 nextChoice = 0;
  for (UInt32 i = 0; i < nPopulation; ++i)
  {
    if (r.getUInt32(nPopulation - i) < (nChoices - nextChoice))
    {
      *choices++ = population[i];
      ++nextChoice;
      if (nextChoice == nChoices)
      {
        break;
      }
    }
  }

}

// Print the appropriate row of the sparse matrix
void printRow(UInt32 row, SparseMatrix01<UInt, Int> *sm)
{
  if (row >= sm->nRows())
  {
    NTA_THROW << "Row size is too big!";
  }
  UInt32 nnz = sm->nNonZerosRow(row);
  vector<UInt> indices;
  indices.resize(nnz);
  sm->getRowSparse(row, indices.begin());
  cout << row << " : ";
  for (int i= 0; i < nnz; i++)
  {
    cout << indices[i] << " ";
  }
  cout << "\n";
}

// Takes a directory containing a category of MNIST images, the number of
// images, and a binarization threshold. Read each image into a row of the
// sparse matrix sm. The image is binarized to 0/1 pixel values using the
// threshold. The SparseMatrix must have 28*28 columns (corresponding to MNIST
// image size)
void readMNISTImages(const char* mnistPath,
                     Int numImages, Int threshold,
                     SparseMatrix01<UInt, Int> *sm)
{
  char fileName[256];
  vector<UInt> activePixels;
  int totalActivePixels = 0;

  // Loop through each image, create filename, and read binarized version
  vector<UInt> population;
  for (int i= 0; i < numImages; i++)
  {
    sprintf(fileName, "%s/%06d.txt", mnistPath, i);
    ifstream f(fileName, ifstream::in);

    if (f)
    {
      Int w, h;
      f >> w >> h;

      // Read in pixels and binarize
      Int p;
      for (int j= 0; j < 28*28; j++)
      {
        f >> p;
        if ( p >= threshold )
        {
          activePixels.push_back(j);
        }
      }
      sm->addRow(activePixels.size(), activePixels.begin());

      totalActivePixels += activePixels.size();

//    cout << "Found " << activePixels.size() << " pixels:\n   ";
//    for (int j= 0; j < activePixels.size(); j++)
//    {
//        cout << activePixels[j] << " ";
//    }
//    cout << "\n";

      f.close();
      activePixels.clear();
    }
    else
    {
      cerr << "File " << fileName << " could not be opened!\n";
    }


  }

  cout << "Average number of active pixels= "
       << ((float) totalActivePixels) / sm->nRows() << "\n";

}

// Run inference on the k'th row of dataSet using the trained dendritic
// model. Return the number of dendrites whose overlap with the k'th row is
// >= threshold.
int runInferenceOnPattern(int row, int threshold,
           SparseMatrix01<UInt, Int> *dataSet,
           SparseMatrix01<UInt, Int> *dendrites)
{
  // Create vector to hold resulting overlaps
  vector<UInt> overlaps;
  overlaps.resize(dendrites->nRows(), 0);

  // Create a dense version of pattern
  vector<UInt> denseX;
  denseX.resize(dataSet->nCols(), 0);
  dataSet->getRow(row, denseX.begin());

  dendrites->rightVecProd(denseX.begin(), overlaps.begin());

  int nMatches = 0;
  for (int i = 0; i < overlaps.size(); i++)
  {
    int nnz = dendrites->nNonZerosRow(i);
    int t = min((int)(0.95*nnz), threshold);
    if (overlaps[i] >= t)
    {
//      if (overlaps[i] == nnz) cout << "Overlap of " << i << " = " << overlaps[i] << "\n";
      nMatches++;
    }
  }

  return nMatches;
}

// Classify pattern the given row by running through each dendrite model
// The dendrite model with the most matches wins
int classifyPattern(int row, int threshold,
           SparseMatrix01<UInt, Int> *dataSet,
           std::vector< SparseMatrix01<UInt, Int> * > &dendrites)
{
  int bestClass = -1;
  int bestOverlap = -1;
  for (int i=0; i < dendrites.size(); i++)
  {
    int matches = runInferenceOnPattern(row, threshold, dataSet, dendrites[i]);
//    cout << "For model " << i << ",  matches=" << matches<< "\n";
    if (matches > bestOverlap)
    {
      bestOverlap = matches;
      bestClass= i;
    }
  }

//  cout << "bestOverlap=" << bestOverlap << "\n";
  return bestClass;
}

// Go through all training examples of class k. Create a set of dendrites
// that randomly sample for that class.
void trainDendrites(int k, int nSynapses,
           std::vector< SparseMatrix01<UInt, Int> * > &trainingSet,
           std::vector< SparseMatrix01<UInt, Int> * > &dendrites,
           Random &r)
{
  for (int i=0; i<trainingSet[k]->nRows(); i++)
  {
    int nnz = trainingSet[k]->nNonZerosRow(i);
    UInt32 synapsesToCreate = nSynapses;
    if (nnz < nSynapses) {
      synapsesToCreate = nnz;
//      cout << "For class " << k << " training image " << i << " has " << nnz
//           << " non-zeros\n";
    }

    // Randomly sample from the non-zero pixels in the image
    vector<UInt> synapseIndices;
    synapseIndices.resize(synapsesToCreate);
    sample(trainingSet[k], i, synapseIndices.begin(), synapsesToCreate, r);

    // Add this to the k'th dendrites model
    dendrites[k]->addRow(synapseIndices.size(), synapseIndices.begin());

    // Verify by getting the last row and printing it out
    // printRow(dendrites[k]->nRows()-1, dendrites[k]);
  }
}

// Read test/training images from top-level directory and return number read
int readImages(int *numImages, const char *path,
                std::vector< SparseMatrix01<UInt, Int> * > &images)
{
  // Read in this percentage of all samples (for faster debugging only)
  Real samplingFactor = 1.0;

  int n = 0;
  for (int i=0; i<10; i++)
  {
    int numImagesToRead = (int)(samplingFactor*numImages[i]);
    char dirName[256];
    sprintf(dirName, path, i);
    readMNISTImages(dirName, numImagesToRead, 64, images[i]);
//    cout << "Read in " << images[i]->nRows() << " images from "
//         << dirName << "\n";
    n += images[i]->nRows();
  }

  return n;
}

// Classify the dataset using a trained dendrite model and the
// given threshold, and report accuracy
void classifyDataset(
           int threshold,
           std::vector< SparseMatrix01<UInt, Int> * > &dataSet,
           std::vector< SparseMatrix01<UInt, Int> * > &dendrites)
{
  int numCorrect = 0, numInferences = 0;
  for (int category=0; category < 10; category++)
  {
    int numCorrectClass = 0;
    for (int k= 0; k<dataSet[category]->nRows(); k++)
    {
      int bestClass = classifyPattern(k, threshold, dataSet[category], dendrites);
      if (bestClass == category)
      {
        numCorrect++;
        numCorrectClass++;
      }
      numInferences++;
    }

    cout << "Category=" << category
         << ", num examples=" << dataSet[category]->nRows()
         << ", pct correct="
         << ((float) numCorrectClass)/dataSet[category]->nRows()
         << "\n";

  }

  cout << "Overall accuracy = " << (100.0 * numCorrect)/numInferences << "%\n";
}


// Run the whole MNIST example.
void runMNIST()
{
  //////////////////////////////////////////////////////
  //
  // Initialize the sparse matrix data structures. Our classifier will be a set
  // of dendrites. dendrites[k] will contain a list of dendrites trained on
  // class k.
  std::vector< SparseMatrix01<UInt, Int> * > dendrites;
  std::vector< SparseMatrix01<UInt, Int> * > trainingSet;
  std::vector< SparseMatrix01<UInt, Int> * > testSet;
  Random r(42);

  for (int i= 0; i<10; i++)
  {
    trainingSet.push_back( new SparseMatrix01<UInt, Int>(28*28, 1));
    dendrites.push_back( new SparseMatrix01<UInt, Int>(28*28, 1));
    testSet.push_back( new SparseMatrix01<UInt, Int>(28*28, 1));
  }

  //////////////////////////////////////////////////////
  //
  // Read in the given number of training and test sets
  int trainingImages[] = {
    5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949
  };
  int testImages[] = { 980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009 };
  int numImages = readImages(trainingImages,
             "../image_test/mnist_extraction_source/training/%d", trainingSet);
  cout << "Read in " << numImages << " total images\n";

  int numTestImages = readImages(testImages,
             "../image_test/mnist_extraction_source/testing/%d", testSet);
  cout << "Read in " << numTestImages << " total test images\n";


  //////////////////////////////////////////////////////
  //
  // Create trained model by randomly sampling from training images
  cout << "Training dendrite model with 100 synapses per dendrite.\n";
  for (int k= 0; k<10; k++)
  {
    trainDendrites(k, 100, trainingSet, dendrites, r);
  }


  //////////////////////////////////////////////////////
  //
  // Classify the data sets and compute accuracy
  cout << "Running classification with a bunch of different thresholds.\n";
  for (int threshold = 68; threshold <= 68; threshold+= 2)
  {
    cout << "\nUsing threshold = " << threshold << "\n";
//    cout << "Training set:";
//    classifyDataset(threshold, trainingSet, dendrites);
    cout << "Test set: ";
    classifyDataset(threshold, testSet, dendrites);
  }

}

// Run the trials!  Currently need to hard code the specific trial you are
// about to run.
int main(int argc, char * argv[])
{
  runMNIST();
}

