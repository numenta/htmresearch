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

#include "dendrite_classifier2.hpp"

using namespace std;
using namespace nupic;


//////////////////////////////////////////////////////
//
// Construction and destruction


DendriteClassifier::DendriteClassifier(int seed, int numClasses, int inputSize,
  int nPrototypesPerClass)
{
  numClasses_ = numClasses;
  rng_ = Random(seed);

  for (int i= 0; i<numClasses; i++)
  {
    dendrites_.push_back( new SparseMatrix01<UInt, Int>(inputSize, 1));
  }

  nPrototypesPerClass_ = nPrototypesPerClass;
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

// This top-level routine does steps 1 and 2 for the entire training set.
void DendriteClassifier::trainDataset(int nSynapses, int threshold,
    std::vector< SparseMatrix01<UInt, Int> * > &trainingSet)
{
  cout << "Training KNN with " << nSynapses << " synapses per dendrite and "
       << nPrototypesPerClass_ << " prototypes per class.\n";

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
  // The KNN will contain one row for each pattern in the training set
  train(threshold, trainingSet);

  cout << "KNN rows=" << knn_->nRows() << " category rows="
       << knn_categories_.size() << "\n";
}

// This routine does Step 3.
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
//      cout << "\nClassifying pattern: " << k << "\n";
      int bestClass = classifyPattern(k, threshold, dataSet[category]);
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
         << ((float) numCorrectClass)/dataSet[category]->nRows() << "\n";
  }

  cout << "\nOverall accuracy = " << (100.0 * numCorrect)/numInferences << "%\n";
}


//
// This routine does step 1. It creates nPrototypesPerClass_ dendrites by
// randomly sampling from trainingSet[k]
void DendriteClassifier::createRandomlySampledDendrites(int category,
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

// Given the p'th pattern in the dataSet, run inference using all the
// dendrites and the given threshold. inferenceNonZeros will contain the
// concatenated list of the responses from all the dendrites.
void DendriteClassifier::inference(int p, int threshold,
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
void DendriteClassifier::train(int threshold,
           std::vector< SparseMatrix01<UInt, Int> * > &dataSet)
{
  int nPatterns = 0;
  for (int category=0; category < dataSet.size(); category++)
  {
    nPatterns += dataSet[category]->nRows();
    for (int p= 0; p<dataSet[category]->nRows(); p++)
    {
      // Run inference on the p'th pattern in this category
      vector<UInt> inferenceNonZeros;
      inference(p, threshold, dataSet[category], inferenceNonZeros);

      // Now we can add this vector to the KNN
      knn_->addRow(inferenceNonZeros.size(), inferenceNonZeros.begin());
      knn_categories_.push_back(category);
    }
  }

  // Sanity checks. Ensure there is one row for each pattern in the dataSet
  if (knn_->nRows() != nPatterns)
  {
    NTA_THROW << "KNN does not have " << nPatterns
              << " rows. It has " << knn_->nRows() << "patterns.\n";
  }
  if (knn_categories_.size() != nPatterns)
  {
    NTA_THROW << "KNN category list does not have " << nPatterns
              << " rows. It has " << knn_categories_.size() << "patterns.\n";
  }
}


// For the given pattern (row in the dataset), run the image through the
// dendrites, check the set of active dendrites against the stored set by doing
// a dot product. Choose category corresponding to highest dot product.
int DendriteClassifier::classifyPattern(int p, int threshold,
           SparseMatrix01<UInt, Int> *dataSet)
{
  // Run inference through the dendritic model to get the overall dendrite
  // response for this pattern.
  vector<UInt> inferenceNonZeros;
  inference(p, threshold, dataSet, inferenceNonZeros);

//  cout << "Number of dendrites with non-zero responses: " << inferenceNonZeros.size() << "\n";

  // Now we need to check which patterns in the KNN have highest overlap
  // with this pattern's vector.

  // Create a dense version of dendritic response vector
//  cout << "Non-zero dendrites: ";
  vector<UInt> denseX;
  denseX.resize(numClasses_*nPrototypesPerClass_, 0);
  for (int i=0; i<inferenceNonZeros.size(); i++)
  {
//    cout << inferenceNonZeros[i] << " ";
    denseX[inferenceNonZeros[i]] = 1;
  }
//  cout << "\n";

  // Do dot product of KNN with dendrite response vector
  vector<UInt> overlaps;
  overlaps.resize(knn_->nRows(), 0);
  knn_->rightVecProd(denseX.begin(), overlaps.begin());

  // Find top matches
  int bestClass = -1;
  UInt bestOverlap = 0;
  for (int i = 0; i < overlaps.size(); i++)
  {
//    cout << "    Overlap with stored pattern " << i << " with category: "
//         << knn_categories_[i] << " is " << overlaps[i] << "\n";
    if (overlaps[i] > bestOverlap)
    {
      bestOverlap = overlaps[i];
      bestClass = knn_categories_[i];
    }
  }

//  cout << "bestClass=" << bestClass << " bestOverlap=" << bestOverlap << "\n";
  return bestClass;
}

