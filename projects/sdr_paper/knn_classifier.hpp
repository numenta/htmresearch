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
#include <nupic/math/NearestNeighbor.hpp>
#include <nupic/types/Types.hpp>
#include <nupic/utils/Random.hpp>

using namespace std;
using namespace nupic;


class KNNClassifier {
  public:

    KNNClassifier(int numClasses=10, int inputSize=784);
    virtual ~KNNClassifier();

    // Go through all training examples for each class. For each class, create
    // a set of dendrites that randomly sample from that class.
    void trainDataset(std::vector< SparseMatrix01<UInt, Int> * > &trainingSet);

    // Classify the dataset using the given value of k, and report accuracy
    void classifyDataset(int k,
                         std::vector< SparseMatrix01<UInt, Int> * > &dataSet);

    // Go through all training examples of category and store in the KNN.
    void trainClass(int category,
           std::vector< SparseMatrix01<UInt, Int> * > &trainingSet);

    //////////////////////////////////////////////////////
    //
    // Internal variables and methods.
    // Leaving as public for ease of debugging.
    int numClasses_;
    int inputSize_;
    NearestNeighbor<SparseMatrix<UInt, Int>> *knn_;
    vector<UInt> knn_categories_;
};
