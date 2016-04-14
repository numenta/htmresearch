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

using namespace std;
using namespace nupic;

//////////////////////////////////////////////////////
//
// Two helper methods to read images

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

  cout << "Avg num of active pixels= "
       << ((float) totalActivePixels) / sm->nRows();

}

// Read test/training images from top-level directory and return number read
// samplingFactor allows you to read in this percentage of all samples (for
// faster debugging only)
int readImages(int *numImages, const char *path,
                std::vector< SparseMatrix01<UInt, Int> * > &images,
                Real samplingFactor = 1.0)
{
  int n = 0;
  for (int i=0; i<10; i++)
  {
    int numImagesToRead = (int)(samplingFactor*numImages[i]);
    char dirName[256];
    sprintf(dirName, path, i);
    readMNISTImages(dirName, numImagesToRead, 64, images[i]);
    cout << ", " << images[i]->nRows() << " images in "
         << dirName << "\n";
    n += images[i]->nRows();
  }

  return n;
}

// Create a new dataset that is a noisy version of the provided one
//void createNoisyDataset(
//                std::vector< SparseMatrix01<UInt, Int> * > &dataset,
//                std::vector< SparseMatrix01<UInt, Int> * > &noisyDataset,
//                        float pctNoise,
//                        Random &r
//                        )
//{
//
//}


