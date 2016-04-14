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

#include "dendrite_classifier1.hpp"

using namespace std;
using namespace nupic;


//////////////////////////////////////////////////////
//
// Construction and destruction


DendriteClassifier1::DendriteClassifier1(
  int nPrototypesPerClass, int seed, int numClasses, int inputSize)
{
  numClasses_ = numClasses;
  rng_ = Random(seed);

  for (int i= 0; i<numClasses; i++)
  {
    dendrites_.push_back( new SparseMatrix01<UInt, Int>(inputSize, 1));
  }

  nPrototypesPerClass_ = nPrototypesPerClass;
}

DendriteClassifier1::~DendriteClassifier1()
{
  deleteDendrites_();
}

void DendriteClassifier1::deleteDendrites_() {
  cout << "Should be deleting dendrites, but I'm not!\n";
}

//////////////////////////////////////////////////////
//
// Methods to train the model


// Populate choices with a random selection of nChoices elements from
// the given row in the given sparse matrix. Throws exception when
// nPopulation < nChoices. This function is used to randomly sample synapses
// from an image.
template <typename ChoicesIter>
void DendriteClassifier1::sample(SparseMatrix01<UInt, Int> *sm, UInt32 row,
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
static void printRow(UInt32 row, SparseMatrix01<UInt, Int> *sm)
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


void DendriteClassifier1::trainDataset(int nSynapses,
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
void DendriteClassifier1::trainClass(int k, int nSynapses,
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
void DendriteClassifier1::classifyDataset(
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
int DendriteClassifier1::runInferenceOnPattern(int row, int threshold,
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
int DendriteClassifier1::classifyPattern(int row, int threshold,
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

//
// This routine does step 1. It creates nPrototypesPerClass_ dendrites by
// randomly sampling from trainingSet[k]
void DendriteClassifier1::createRandomlySampledDendrites(int category,
    int nSynapses, std::vector< SparseMatrix01<UInt, Int> * > &trainingSet)
{
  for (int j=0; j<nPrototypesPerClass_; j++)
  {
    // Randomly choose i'th pattern from training vectors for this category
    UInt32 i = rng_.getUInt32(trainingSet[category]->nRows());

    int nnz = trainingSet[category]->nNonZerosRow(i);
    UInt32 synapsesToCreate = nSynapses;
    if (nnz < nSynapses) {
      synapsesToCreate = nnz;
    }
//    cout << "For category " << category << " randomly chosen training image "
//         << i << " has " << nnz << " non-zeros\n";

    // Randomly sample from the non-zero pixels in the image
    vector<UInt> synapseIndices;
    synapseIndices.resize(synapsesToCreate);
    sample(trainingSet[category], i, synapseIndices.begin(), synapsesToCreate);

    // Add this to the k'th dendrites model
    dendrites_[category]->addRow(synapseIndices.size(), synapseIndices.begin());

    // Verify by getting the last row and printing it out
//    cout << "Randomly sampled dendrite: ";
//    printRow(dendrites_[category]->nRows()-1, dendrites_[category]);
//    if (category==1)
//    {
//      cout << "For category " << category << " randomly chosen training image "
//           << i << " has " << nnz << " non-zeros\n";
//      cout << "Randomly sampled dendrite: ";
//      printRow(dendrites_[category]->nRows()-1, dendrites_[category]);
//    }
  }
}
