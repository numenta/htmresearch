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


// Print the given vector/array
template <typename ValueType> void printVector(ValueType &indices, UInt w)
{
  for (UInt i = 0; i < w; i++)
  {
    std::cout << indices[i] << " ";
  }
  std::cout << std::endl;
}

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


// Insert a new image into new_sm that is an occludded version of the image in
// the given row of sm. Occlusion rule: randomly choose noise pct bits from the
// ON bits and turn them off (ala Kolankeh, et al)
void createOccludedImage(int row,
        SparseMatrix01<UInt, Int> *sm,
        SparseMatrix01<UInt, Int> *new_sm,
        float noisePct, Random &r )
{
  UInt32 nnz = sm->nNonZerosRow(row);

  // Get sparse version of the image in this row
  vector<UInt> x;
  x.resize(nnz, 0);
  sm->getRowSparse(row, x.begin());

  // This will hold the image with occlusions
  vector<UInt> xp;

  // Randomly insert 1-noisePct of the active bits in image into the new image
  for (int i=0; i < x.size(); i++)
  {
    if (r.getReal64() > noisePct)
    {
      xp.push_back(x[i]);
    }
  }

  new_sm->addRow(xp.size(), xp.begin());

//  cout << "\nImage before. Size=" << x.size() << " bits: ";
//  printVector(x, x.size());
//  cout << "Image after. Size=" << xp.size() << " bits: ";
//  printVector(xp, xp.size());

}


// Create a new dataset that is a noisy version of the provided one.
// Noise type 1: swap noise of the on bits with off bits
// Noise type 2: occlusion. Randomly choose noise bits from the ON bits and
//               turn them off (ala Kolankeh, et al)
// Note: will clear out noiseDataset and create new SparseMatrix01 instances.
void createNoisyDataset(
        std::vector< SparseMatrix01<UInt, Int> * > &dataset,
        std::vector< SparseMatrix01<UInt, Int> * > &noisyDataset,
        float noisePct, int noiseType, Random &r )
{
  auto numClasses = dataset.size();
  auto numPixels = dataset[0]->nCols();
  noisyDataset.clear();
  for (int category = 0; category < numClasses; category++)
  {
    // Create new empty dataset
    noisyDataset.push_back( new SparseMatrix01<UInt, Int>(numPixels, 1));

    for (int row=0; row < dataset[category]->nRows(); row++)
    {
      if (noiseType==2)
      {
        createOccludedImage(row, dataset[category], noisyDataset[category],
                            noisePct, r);
        // Output number of ON bits in image before and after
//        cout << "(" << dataset[category]->nNonZerosRow(row) << ","
//                    << noisyDataset[category]->nNonZerosRow(row) << ") ";
      }
      else
      {
        NTA_THROW << "Unsupported noise type!\n";
      }
    }

    cout << "\n";

    // Sanity check
    if (noisyDataset[category]->nRows() != dataset[category]->nRows())
    {
      NTA_THROW << "Incorrect number of rows. Noise dataset has "
                << noisyDataset[category]->nRows() << " rows. Expected: "
                << dataset[category]->nRows() << "\n";
    }
  }

}


