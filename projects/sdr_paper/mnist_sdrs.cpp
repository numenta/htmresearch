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

// Takes a directory containing a category of MNIST images, the number of images,
// and a binarization threshold.
// Read each image into a row of the sparse matrix sm. The image is binarized
// to 0/1 pixel values using the threshold.
// The SparseMatrix must have 28*28 columns (corresponding to MNIST image size)
void readMNISTImages(const char* mnistPath,
                     Int numImages, Int threshold,
                     SparseMatrix01<UInt, Int> *sm)
{
  char fileName[256];
  vector<UInt> activePixels;

  // Loop through each image
  // Create file name for each image
  vector<UInt> population;
  for (int i= 0; i < numImages; i++)
  {
    sprintf(fileName, "%s/%06d.txt", mnistPath, i);
    ifstream f(fileName, ifstream::in);

    if (f)
    {
      Int w, h;
      f >> w >> h;

//      printf("Opening %s, w=%d, h=%d\n", fileName, w, h);

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

}

// Run the whole MNIST example.
void runMNIST()
{
  int trainingImages[] = {
    5923, 6742, 5957, 6130, 5841, 5420, 5917, 6264, 5850, 5948
  };
  std::vector< SparseMatrix01<UInt, Int> * > trainingSet;

  // Training set
  SparseMatrix01<UInt, Int>
    category0(28*28, 1),
    category1(28*28, 1),
    category2(28*28, 1),
    category3(28*28, 1),
    category4(28*28, 1),
    category5(28*28, 1),
    category6(28*28, 1),
    category7(28*28, 1),
    category8(28*28, 1),
    category9(28*28, 1)
  ;

  for (int i= 0; i<10; i++)
  {
    trainingSet.push_back( new SparseMatrix01<UInt, Int>(28*28, 1));
  }

  // Training set
  std::vector< SparseMatrix01<UInt, Int> * > testSet;
//  SparseMatrix01<UInt, Int>
//    category0(28*28, 1),
//    category1(28*28, 1),
//    category2(28*28, 1),
//    category3(28*28, 1),
//    category4(28*28, 1),
//    category5(28*28, 1),
//    category6(28*28, 1),
//    category7(28*28, 1),
//    category8(28*28, 1),
//    category9(28*28, 1)
//  ;

  int numImages = 0;
  for (int i=0; i<10; i++)
  {
    char dirName[256];
    sprintf(dirName, "../image_test/mnist_extraction_source/training/%d", i);
    readMNISTImages(dirName, trainingImages[i], 128, trainingSet[i]);
    cout << "Read in " << trainingSet[i]->nRows() << " images from "
         << dirName << "\n";
    numImages += trainingSet[i]->nRows();
  }
  cout << "Read in " << numImages << " total images\n";


//  readMNISTImages("../image_test/mnist_extraction_source/training/1",
//                  6742, 128, trainingSet[1]);
//  cout << "Read in " << category1.nRows() << " images\n";
//
//  readMNISTImages("../image_test/mnist_extraction_source/training/2",
//                  5957, 128, category2);
//  cout << "Read in " << category2.nRows() << " images\n";


}



