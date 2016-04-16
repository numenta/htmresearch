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

#include "binary_algorithms.hpp"

using namespace std;
using namespace nupic;

// populate choices with a random selection of nChoices elements from
// population. throws exception when nPopulation < nChoices
// templated functions must be defined in header
template <typename PopulationIter, typename ChoicesIter>
void sample(PopulationIter population, UInt32 nPopulation,
            ChoicesIter choices, UInt32 nChoices, Random &r)
{
  if (nChoices == 0)
  {
    return;
  }
  if (nChoices > nPopulation)
  {
    NTA_THROW << "population size must be greater than number of choices";
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

// Estimate the confidence bounds. This assumes a binomial distribution and
// computes the +- for a 95% confidence bound on a probability prob computed
// from the given number of trials.
Real estimateBounds(Real prob, Int trials)
{
  const Real z = 1.96;  // 95% from normal distribution
  Real stddev = sqrt((prob * (1.0 - prob)) / (Real)trials);
  return z * stddev;
}

// Given a set of patterns, a new pattern x, and a match threshold theta,
// return the number of matching vectors
int numMatches(SparseMatrix01<UInt, Int> &patterns,
               vector<UInt> &x, UInt theta)
{
  // Create a dense version of x
  vector<UInt> denseX;
  denseX.resize(patterns.nCols(), 0);
  for (auto it = x.begin(); it != x.end(); ++it)
  {
    denseX[*it] = 1;
  }

  // Create empty overlaps vector
  vector<UInt> overlaps;
  overlaps.resize(patterns.nRows(), 0);

  patterns.rightVecProd(denseX.begin(), overlaps.begin());

  int nMatches = 0;
  for (int i = 0; i < overlaps.size(); i++)
  {
    if (overlaps[i] >= theta) nMatches++;
  }

  return nMatches;
}

// Given a union of patterns, a new pattern x, and a match threshold theta,
// return the number of matching vectors
int unionNumMatches(set<UInt> &patterns, vector<UInt> &x, UInt theta)
{
  int nMatches = 0;
  for (auto it = x.begin(); it != x.end(); it++)
  {
    if (patterns.find(*it) != patterns.end()) nMatches++;
  }

  if (nMatches >= theta)
    return 1;
  else
    return 0;
}

// Change exactly `noise` of the ON bits from x and put the result in xp.
void addNoise(const vector<UInt>& x, vector<UInt>& xp, UInt n, UInt w,
              UInt noise, Random& r)
{
  // Create a population that does not include the original bits
  vector<UInt32> addOptions;
  for (Int i=0; i < n; i++) addOptions.push_back(i);
  // Iterate in reverse order so indices in addOptions don't change
  for (Int i=w-1; i >= 0; --i) addOptions.erase(addOptions.begin() + x[i]);
  NTA_ASSERT(addOptions.size() == (n - w));

//  cout << "addOptions: ";
//  printSparseIndices(addOptions, addOptions.size());

//  UInt nAdded = r.getUInt32(noise + 1);
//  UInt nRemoved = noise - nAdded;
//
//  xp.resize(w + nAdded - nRemoved);
//  sample(x.begin(), w, xp.begin(), w - nRemoved, r);
//  sample(addOptions.begin(), addOptions.size(),
//         xp.begin() + w - nRemoved, nAdded, r);

  xp.resize(w);
  // Randomly sample w-noise bits from x that we will keep
  sample(x.begin(), w, xp.begin(), w - noise, r);

  // Randomly sample noise bits from addOptions that we will put into xp
  sample(addOptions.begin(), addOptions.size(),
         xp.begin() + w - noise, noise, r);

}

// Create M different vectors, each with w random bits on, and add them to sm.
// Each vectors will have sm.ncols()
void createRandomVectors(Int M, Int w, SparseMatrix01<UInt, Int> &sm,
                         Random &r, int verbosity=0)
{
  vector<UInt> population;
  for (int i= 0; i < sm.nCols(); i++) population.push_back(i);

  vector<UInt> activeBits;
  activeBits.resize(w);

  if (verbosity > 0)
  {
    cout << "Creating " << M << " random vectors with " << w << " bits on.\n";
  }
  for (Int m=0; m < M; m++)
  {
    // Randomly sample from columns
    sample(population.begin(), sm.nCols(), activeBits.begin(), w, r);
    sm.addRow(activeBits.size(), activeBits.begin());

    if (verbosity > 1)
    {
      cout << m << ":";
      for (Int i = 0; i < w; i++)
      {
        cout << activeBits[i] << " ";
      }
      cout << endl;
    }
  }
}

// Create M different vectors, each with w random bits on, and union them
// with s.
void unionRandomVectors(UInt M, UInt n, UInt w, set<UInt>& s,
                        Random &r, int verbosity=0)
{
  vector<UInt> population;
  for (int i= 0; i < n; i++) population.push_back(i);

  vector<UInt> activeBits;
  activeBits.resize(w);

  if (verbosity > 0)
  {
    cout << "Creating " << M << " random vectors with " << w << " bits on.\n";
  }
  for (Int m=0; m < M; m++)
  {
    // Randomly sample from columns
    sample(population.begin(), n, activeBits.begin(), w, r);
    for (UInt i = 0; i < w; i++)
    {
      s.insert(activeBits[i]);
    }

    if (verbosity > 1)
    {
      cout << m << ":";
      for (Int i = 0; i < w; i++)
      {
        cout << activeBits[i] << " ";
      }
      cout << endl;
    }
  }
}

