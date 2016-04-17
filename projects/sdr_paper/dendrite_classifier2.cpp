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
    weights_.push_back( new vector<Real>);
  }

  nDendritesPerClass_ = nPrototypesPerClass;

}

DendriteClassifier::~DendriteClassifier()
{
  deleteDendrites_();
}

void DendriteClassifier::deleteDendrites_() {
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

// Print the given vector/array
template <typename ValueType> void printVector(ValueType &indices, UInt w)
{
  for (UInt i = 0; i < w; i++)
  {
    std::cout << indices[i] << " ";
  }
  std::cout << std::endl;
}


// This top-level routine does steps 1 and 2 for the entire training set.
void DendriteClassifier::trainDataset(int nSynapses, int threshold,
    std::vector< SparseMatrix01<UInt, Int> * > &trainingSet,
    bool useDefaultWeights)
{
  cout << "Training model with " << nSynapses << " synapses per dendrite and "
       << nDendritesPerClass_ << " dendrites per class.\n";

  // Create randomly sampled dendrites for each class. There will be
  // nDendritesPerClass_ in dendrites[category] after this routine is finished.
  for (int category = 0; category < numClasses_; category++)
  {
    createRandomlySampledDendrites(category, nSynapses, trainingSet);
  }

  // Sanity check
  for (int i=0; i < numClasses_; i++)
  {
    if ( dendrites_[i]->nRows() != nDendritesPerClass_)
    {
      NTA_THROW << "Dendrite model for class " << i << " does not have"
                << " sufficient prototypes\n";
    }
//    for (int j=0; j < nDendritesPerClass_; j++)
//    {
//      cout << "Category " << i << " prototype " << j << " non-zeros="
//           << dendrites_[i]->nNonZerosRow(j) << "\n";
//    }
  }

  // We have two options for setting the weights.
  if (useDefaultWeights)
  {
    //  Initialize using default weights
    initializeDefaultWeights();

    // Output A matrix so that we can train the least squares weights
    saveModelResponses(threshold, trainingSet);
  }
  else
  {
    // Read in the saved least squares weights
    readLeastSquaresWeights();
  }

}


// This routine does Step 3.
void DendriteClassifier::classifyDataset(
           int threshold,
           std::vector< SparseMatrix01<UInt, Int> * > &dataSet)
{
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
// This routine does step 1. It creates nDendritesPerClass_ dendrites by
// randomly sampling from trainingSet[k]
void DendriteClassifier::createRandomlySampledDendrites(int category,
    int nSynapses, std::vector< SparseMatrix01<UInt, Int> * > &trainingSet)
{
  for (int j=0; j<nDendritesPerClass_; j++)
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
      int t = min((int)(0.8*nnz), threshold);
      int index = nDendritesPerClass_*k + i;

//      if ( (p==0) && (index == 2) )
//      {
//        cout << "Image non-zeros= " << dataSet->nNonZerosRow(p) << " ";
//        cout << "Dendrite 2, nnz= " << nnz << " t= " << t
//             << " overlap= " << overlaps[i] << "\n";
//      }
//
      if (overlaps[i] >= t)
      {
        // Add non-zero elements for this match using a global index for this
        // dendrite model
        inferenceNonZeros.push_back(index);
      }
    }
  }

}


// This routine assumes the dendrite models have been created and outputs
// their responses to each pattern in the dataset to a csv file.
//
// For each category we solve the least squares solution for Aw = Y where:
//  A: M rows, one for each input pattern p
//     nDendritesPerClass_*numClasses_ cols, one for each dendrite model j
//
//  Y: M elements, containing desired label for input pattern i
//
//  W: nDendritesPerClass_*numClasses_ weights for category
//
// This routine will output A into a file for offline least squares training
void DendriteClassifier::saveModelResponses(int threshold,
           std::vector< SparseMatrix01<UInt, Int> * > &dataSet)
{
  string outPath("dendrite_responses.csv");
  ofstream f(outPath);

  dendrite_responses_.clear();
  dendrite_responses_.resize(nDendritesPerClass_*numClasses_);

  // Compute A by running each pattern in the dataset through the entire dataset.
  int nPatterns = 0;
  for (int category=0; category < dataSet.size(); category++)
  {
    nPatterns += dataSet[category]->nRows();

    for (int p= 0; p<dataSet[category]->nRows(); p++)
    {
      // Run inference on the p'th pattern in this category
      vector<UInt> inferenceNonZeros;
      inference(p, threshold, dataSet[category], inferenceNonZeros);

//      cout << "Category: " << category << ", responses: ";
//      printVector(inferenceNonZeros, inferenceNonZeros.size());

      // Save each row to our file
      if (inferenceNonZeros.size()>0)
      {
        for (UInt i = 0; i < inferenceNonZeros.size()-1; i++)
        {
          f << inferenceNonZeros[i] << ",";
          dendrite_responses_[inferenceNonZeros[i]] += 1;
        }
        dendrite_responses_[inferenceNonZeros.size()-1] += 1;
        f << inferenceNonZeros[inferenceNonZeros.size()-1];
      }
      f << std::endl;
    }
  }

  f.close();

//  cout << "Dendrite responses:\n";
//  printVector(dendrite_responses_, dendrite_responses_.size());

}


// For the given pattern (row in the dataset), run the image through the
// dendrites, check the set of active dendrites against the stored set by doing
// a dot product. Choose category corresponding to highest dot product.
int DendriteClassifier::classifyPattern(int p, int threshold,
           SparseMatrix01<UInt, Int> *dataSet)
{
  // Run inference through the dendritic model to get the overall dendrite
  // response vector for this pattern.
  vector<UInt> inferenceNonZeros;
  inference(p, threshold, dataSet, inferenceNonZeros);
//  if (p < 1)
//  {
//    cout << "Pattern " << p << ", number of dendrites with non-zero responses: "
//         << inferenceNonZeros.size() << "\n";
//    printVector(inferenceNonZeros, inferenceNonZeros.size());
//  }

  // Create a dense version of dendritic response vector
  //  cout << "Non-zero dendrites: ";
  vector<UInt> denseX;
  denseX.resize(numClasses_*nDendritesPerClass_, 0);
  for (int i=0; i<inferenceNonZeros.size(); i++)
  {
  //cout << inferenceNonZeros[i] << " ";
    denseX[inferenceNonZeros[i]] = 1;
  }
//  cout << "\n";

  // Now we use a readout classifier to figure out the final category.
  // We compute a dot product of the dendrite responses with the weights
  // associated with each class and pick the highest one.

  // Find top matches
  int bestClass = -1;
  Real maxProduct = -100000000;
  for (int category=0; category < numClasses_; category++)
  {
    // compute dot product with this category's weights
    Real dp = 0.0;
    int j = 0;
    for(auto it = weights_[category]->begin(); it != weights_[category]->end(); ++it)
    {
      dp += denseX[j++]*(*it);
    }

    // Is it the best so far?
    if (dp > maxProduct)
    {
      maxProduct = dp;
      bestClass = category;
    }
  }

//  cout << "bestClass=" << bestClass << " maxProduct=" << maxProduct << "\n";
  return bestClass;
}


void DendriteClassifier::initializeDefaultWeights()
{
  for (int category=0; category < numClasses_; category++)
  {
    weights_[category]->clear();
    weights_[category]->resize(numClasses_*nDendritesPerClass_, 0.0);
    for (int j=0; j<nDendritesPerClass_; j++)
    {
      weights_[category]->at(nDendritesPerClass_*category + j) = 1.0;
    }
  }

  // Sanity check
//  for (int category=0; category < numClasses_; category++)
//  {
//    cout << "Weights for category: " << category << " are:\n";
//    printVector(*weights_[category], weights_[category]->size());
//  }

}


// Read the least squares weights we have saved to a file
void DendriteClassifier::readLeastSquaresWeights()
{
  cout << "Reading in weights from weights.txt\n";
  ifstream f("weights.txt");
  for (int category=0; category < numClasses_; category++)
  {
    int dim;
    f >> dim;
    if (dim != numClasses_*nDendritesPerClass_)
    {
      NTA_THROW << "Weights for category " << category << " do not have the"
                << " correct dimensionality. Got " << dim << " expected "
                << numClasses_*nDendritesPerClass_ << "\n";
    }
    weights_[category]->clear();
    weights_[category]->resize(numClasses_*nDendritesPerClass_, 0.0);
    for (int j=0; j<dim; j++)
    {
      f >> weights_[category]->at(j);
    }
    // Sanity check
//    cout << "Category: " << category << "\nWeights: ";
//    for (int j=0; j<dim; j++)
//    {
//      cout << weights_[category]->at(j) << " ";
//    }
//    cout << "\n";
  }
  f.close();
}



