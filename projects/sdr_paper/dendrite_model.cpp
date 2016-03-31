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

using namespace std;
using namespace nupic;

/*
class DendriteClassifier {
  public:

    DendriteClassifier(int seed=42, int numClasses = 10);
    virtual ~DendriteClassifier();

    // Go through all training examples for each class. For each class, create
    // a set of dendrites that randomly sample from that class.
    void trainDataset(int nSynapses,
          std::vector< SparseMatrix01<UInt, Int> * > &trainingSet);

    void classifyDataset(int threshold,
               std::vector< SparseMatrix01<UInt, Int> * > &dataSet);


    //////////////////////////////////////////////////////
    //
    // Public methods but lower level

    // Go through all training examples of class k. Create a set of dendrites
    // that randomly sample for that class.
    void trainClass(int k, int nSynapses,
           std::vector< SparseMatrix01<UInt, Int> * > &trainingSet);


    int runInferenceOnPattern(int row, int threshold,
               SparseMatrix01<UInt, Int> *dataSet);


    int classifyPattern(int row, int threshold,
               SparseMatrix01<UInt, Int> *dataSet);

    //////////////////////////////////////////////////////
    //
    // Internal variables and methods.
    // Leaving as public for ease of debugging.
    int numClasses_;
    Random rng_;
    std::vector< SparseMatrix01<UInt, Int> * > dendrites_;

    template <typename ChoicesIter>  void sample(SparseMatrix01<UInt, Int> *sm,
          UInt32 row, ChoicesIter choices, UInt32 nChoices);

     // Clear memory
     void deleteDendrites_();


};

*/

//////////////////////////////////////////////////////
//
// Methods to train the model


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


// Choose N random training examples from class k. For each example, create
// a dendrite that randomly samples from that image.
void trainDendrites2(int k, int nSynapses,
           std::vector< SparseMatrix01<UInt, Int> * > &trainingSet,
           std::vector< SparseMatrix01<UInt, Int> * > &dendrites,
           Random &r)
{
  const int nSamples = 10000;
  for (int j=0; j<nSamples; j++)
  {
    // Choose i'th sample randomly with replacement
    UInt32 i = r.getUInt32(trainingSet[k]->nRows());

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

//////////////////////////////////////////////////////
//
// Methods to run inference on the model


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

// Classify the dataset using a trained dendrite model and the
// given threshold, and report accuracy
void classifyDataset(
           int threshold,
           std::vector< SparseMatrix01<UInt, Int> * > &dataSet,
           std::vector< SparseMatrix01<UInt, Int> * > &dendrites)
{
  int numCorrect = 0, numInferences = 0;

  for (int category=0; category < dataSet.size(); category++)
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

    cout << "\nCategory=" << category
         << ", num examples=" << dataSet[category]->nRows()
         << ", pct correct="
         << ((float) numCorrectClass)/dataSet[category]->nRows();
  }

  cout << "\nOverall accuracy = " << (100.0 * numCorrect)/numInferences << "%\n";
}
