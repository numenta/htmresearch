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

#include "DendriteClassifier.hpp"

using namespace std;
using namespace nupic;


//////////////////////////////////////////////////////
//
// Construction and destruction


DendriteClassifier::DendriteClassifier(int seed, int numClasses, int inputSize)
{
  numClasses_ = numClasses;
  rng_ = Random(seed);

  for (int i= 0; i<numClasses; i++)
  {
    dendrites_.push_back( new SparseMatrix01<UInt, Int>(inputSize, 1));
  }

  usingKNN_ = false;
  nPrototypesPerClass_ = 10;
  knn_ = new SparseMatrix01<UInt, Int>(numClasses*nPrototypesPerClass_, 1);
}

DendriteClassifier::~DendriteClassifier()
{
  deleteDendrites_();
}

void DendriteClassifier::deleteDendrites_() {
  cout << "Should be deleting dendrites and knn, but I'm not!\n";
}

//////////////////////////////////////////////////////
//
// Methods to train the model


// Populate choices with a random selection of nChoices elements from
// the given row in the given sparse matrix. Throws exception when
// nPopulation < nChoices. This function is used to randomly sample synapses
// from an image.
template <typename ChoicesIter>
void DendriteClassifier::sample(SparseMatrix01<UInt, Int> *sm, UInt32 row,
            ChoicesIter choices, UInt32 nChoices)
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
    if (rng_.getUInt32(nPopulation - i) < (nChoices - nextChoice))
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


void DendriteClassifier::trainDataset(int nSynapses,
    std::vector< SparseMatrix01<UInt, Int> * > &trainingSet)
{
  // Ensure trainingSet has enough classes
  cout << "Training dendrite model with " << nSynapses
       << " synapses per dendrite.\n";
  for (int category = 0; category < numClasses_; category++)
  {
    trainClass(category, nSynapses, trainingSet);
  }

}

// Go through all training examples of class k. Create a set of dendrites
// that randomly sample for that class.
void DendriteClassifier::trainClass(int k, int nSynapses,
           std::vector< SparseMatrix01<UInt, Int> * > &trainingSet)
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
    sample(trainingSet[k], i, synapseIndices.begin(), synapsesToCreate);

    // Add this to the k'th dendrites model
    dendrites_[k]->addRow(synapseIndices.size(), synapseIndices.begin());

    // Verify by getting the last row and printing it out
    // printRow(dendrites_[k]->nRows()-1, dendrites_[k]);
  }
}


//////////////////////////////////////////////////////
//
// Methods to run inference on the model

// Classify the dataset using a trained dendrite model and the
// given threshold, and report accuracy
void DendriteClassifier::classifyDataset(
           int threshold,
           std::vector< SparseMatrix01<UInt, Int> * > &dataSet)
{
  // TODO: Verify dataSet has no more than numClasses categories.

  int numCorrect = 0, numInferences = 0;

  for (int category=0; category < dataSet.size(); category++)
  {
    int numCorrectClass = 0;
    for (int k= 0; k<dataSet[category]->nRows(); k++)
    {
      int bestClass = classifyPattern(k, threshold, dataSet[category]);
      if (bestClass == category)
      {
        numCorrect++;
        numCorrectClass++;
      }
      numInferences++;
    }

    cout << "\nCategory=" << category
         << ", num examples=" << dataSet[category]->nRows()
         << ", pct correct="
         << ((float) numCorrectClass)/dataSet[category]->nRows();
  }

  cout << "\nOverall accuracy = " << (100.0 * numCorrect)/numInferences << "%\n";
}


// Run inference on the k'th row of dataSet using the trained dendritic
// model. Return the number of dendrites whose overlap with the k'th row is
// >= threshold.
int DendriteClassifier::runInferenceOnPattern(int row, int threshold,
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
int DendriteClassifier::classifyPattern(int row, int threshold,
           SparseMatrix01<UInt, Int> *dataSet)
{
  int bestClass = -1;
  int bestOverlap = -1;
  for (int i=0; i < dendrites_.size(); i++)
  {
    int matches = runInferenceOnPattern(row, threshold, dataSet, dendrites_[i]);
//  cout << "The model for class " << i << ",  has " << matches<< " matches\n";
    if (matches > bestOverlap)
    {
      bestOverlap = matches;
      bestClass= i;
    }
  }

//  cout << "bestOverlap=" << bestOverlap << "\n\n";
  return bestClass;
}


//////////////////////////////////////////////////////////////////////
//
// Classification scheme 2
//
// A second KNN like classifier using dendrites:
//

// This top-level routine does steps 1 and 2 for the entire training set.
void DendriteClassifier::trainDatasetKNN(int nSynapses, int threshold,
    std::vector< SparseMatrix01<UInt, Int> * > &trainingSet)
{
  // TODO: Ensure trainingSet has enough classes
  cout << "Training KNN with " << nSynapses << " synapses per dendrite.\n";

  // Create randomly sampled dendrites for each class. There will be
  // nPrototypesPerClass_ in dendrites[category] after this routine is finished.
  for (int category = 0; category < numClasses_; category++)
  {
    createRandomlySampledDendrites(category, nSynapses, trainingSet);
  }

  // Sanity check
  for (int i=0; i < numClasses_; i++)
  {
    if ( dendrites_[i]->nRows() != nPrototypesPerClass_)
    {
      NTA_THROW << "Dendrite model for class " << i << " does not have"
                << " sufficient prototypes\n";
    }
//    for (int j=0; j < nPrototypesPerClass_; j++)
//    {
//      cout << "Category " << i << " prototype " << j << " non-zeros="
//           << dendrites_[i]->nNonZerosRow(j) << "\n";
//    }
  }

  // Train KNN on each class using the randomly sampled dataset
  trainKNN(threshold, trainingSet);

  cout << "KNN rows=" << knn_->nRows() << " category rows="
       << knn_categories_.size() << "\n";
}

//
// This routine does step 1.
void DendriteClassifier::createRandomlySampledDendrites(int k, int nSynapses,
           std::vector< SparseMatrix01<UInt, Int> * > &trainingSet)
{
  for (int j=0; j<nPrototypesPerClass_; j++)
  {
    // Choose i'th sample randomly with replacement
    UInt32 i = rng_.getUInt32(trainingSet[k]->nRows());

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
    sample(trainingSet[k], i, synapseIndices.begin(), synapsesToCreate);

    // Add this to the k'th dendrites model
    dendrites_[k]->addRow(synapseIndices.size(), synapseIndices.begin());

    // Verify by getting the last row and printing it out
    // printRow(dendrites_[k]->nRows()-1, dendrites_[k]);
  }
}

// Given the p'th pattern in the dataSet, run inference using all the
// dendrites and the given threshold. inferenceNonZeros will contain the
// concatenated list of the responses from all the dendrites.
void DendriteClassifier::inferenceForKNN(int p, int threshold,
      SparseMatrix01<UInt, Int> *dataSet,
      vector<UInt> &inferenceNonZeros)
{
  // Create a dense version of p'th pattern in this category
  vector<UInt> denseX;
  denseX.resize(dataSet->nCols(), 0);
  dataSet->getRow(p, denseX.begin());

  // We will run dendrite model for each class and create concatenated vector
  inferenceNonZeros.clear();
  for (int k=0; k < numClasses_; k++)
  {
    // Create temp vector to hold resulting overlaps
    vector<UInt> overlaps;
    overlaps.resize(dendrites_[k]->nRows(), 0);

    dendrites_[k]->rightVecProd(denseX.begin(), overlaps.begin());

    for (int i = 0; i < overlaps.size(); i++)
    {
      int nnz = dendrites_[k]->nNonZerosRow(i);
      int t = min((int)(0.95*nnz), threshold);
      if (overlaps[i] >= t)
      {
        // Add non-zero elements for this match
        int index = nPrototypesPerClass_*k + i;
        inferenceNonZeros.push_back(index);
      }
    }
  }

}


// This routine does step 2: run each training pattern through the dendrites and
// store the resulting set of dendritic activity in the KNN.
void DendriteClassifier::trainKNN(int threshold,
           std::vector< SparseMatrix01<UInt, Int> * > &dataSet)
{
  for (int category=0; category < dataSet.size(); category++)
  {
    for (int p= 0; p<dataSet[category]->nRows(); p++)
    {
      // Run inference on the p'th pattern in this category
      vector<UInt> inferenceNonZeros;
      inferenceForKNN(p, threshold, dataSet[category], inferenceNonZeros);

      // Now we can add this vector to the KNN
      knn_->addRow(inferenceNonZeros.size(), inferenceNonZeros.begin());
      knn_categories_.push_back(category);
    }
  }
}


// For the given pattern (row in the dataset), run the image through the
// dendrites, check the set of active dendrites against the stored set by doing
// a dot product. Choose category corresponding to highest dot product.
int DendriteClassifier::classifyPatternKNN(int p, int threshold,
           SparseMatrix01<UInt, Int> *dataSet)
{
  // Run inference through the dendritic model
  vector<UInt> inferenceNonZeros;
  inferenceForKNN(p, threshold, dataSet, inferenceNonZeros);

  // Now we need to check with patterns in the KNN have highest overlap
  // with this pattern's vector.

  // Create a dense version of pattern
  vector<UInt> denseX;
  denseX.resize(numClasses_*nPrototypesPerClass_, 0);
  for (int i=0; i<inferenceNonZeros.size(); i++)
  {
    denseX[inferenceNonZeros[i]] = 1;
  }

  // Do dot product of KNN with inference vector
  vector<UInt> overlaps;
  overlaps.resize(numClasses_*nPrototypesPerClass_, 0);
  knn_->rightVecProd(denseX.begin(), overlaps.begin());

  // Find top matches
  int bestClass = -1;
  int bestOverlap = -1;
  for (int i = 0; i < overlaps.size(); i++)
  {
    if (overlaps[i] >= bestOverlap)
    {
      bestOverlap = overlaps[i];
      bestClass = knn_categories_[i];
    }
  }

  cout << "bestClass=" << bestClass << " bestOverlap=" << bestOverlap << "\n";
  return bestClass;
}

// This routine does Step 3.
void DendriteClassifier::classifyDatasetKNN(
           int threshold,
           std::vector< SparseMatrix01<UInt, Int> * > &dataSet)
{
  // TODO: Verify dataSet has no more than numClasses categories.

  int numCorrect = 0, numInferences = 0;

  for (int category=0; category < dataSet.size(); category++)
  {
    int numCorrectClass = 0;
    for (int k= 0; k<dataSet[category]->nRows(); k++)
    {
      int bestClass = classifyPatternKNN(k, threshold, dataSet[category]);
      if (bestClass == category)
      {
        numCorrect++;
        numCorrectClass++;
      }
      numInferences++;
    }

    cout << "\nCategory=" << category
         << ", num examples=" << dataSet[category]->nRows()
         << ", pct correct="
         << ((float) numCorrectClass)/dataSet[category]->nRows();
  }

  cout << "\nOverall accuracy = " << (100.0 * numCorrect)/numInferences << "%\n";
}