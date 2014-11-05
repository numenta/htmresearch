/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2013, Numenta, Inc.  Unless you have an agreement
 * with Numenta, Inc., for a separate license for this software code, the
 * following terms and conditions apply:
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see http://www.gnu.org/licenses.
 *
 * http://numenta.org/licenses/
 * ----------------------------------------------------------------------
 */


#include <iostream>
#include <assert.h>
#include <nta/math/Math.hpp>
#include <string>
#include <vector>

#include <nta/utils/Random.hpp>
#include <nta/math/SparseMatrix.hpp>
#include <nta/math/SparseMatrix01.hpp>

using namespace std;
using namespace nta;

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

// Given a set of patterns, a new pattern x, and a match threshold theta,
// return the number of matching vectors
int numMatches(SparseMatrix01<UInt, Int> &patterns,
              vector<UInt> &x, UInt theta)
{
  // Create a dense version of x
  vector<UInt> denseX;
  denseX.resize(patterns.nCols(), 0);
  for (vector<UInt>::iterator it = x.begin(); it != x.end(); ++it)
  {
    denseX[*it] = 1;
  }

  // Create empty overlaps vector
  vector<UInt> overlaps;
  overlaps.resize(patterns.nRows(), 0);

  patterns.rightVecProd(denseX.begin(), overlaps.begin());

  int numMatches = 0;
  for (int i= 0; i < overlaps.size(); i++)
  {
    if (overlaps[i] >= theta) numMatches++;
  }

  return numMatches;
}


//
// Create M different vectors, each with w random bits on, and add them to sm.
// Each vectors will have sm.ncols()
//
void createRandomVectors(Int M, Int w, SparseMatrix01<UInt, Int> &sm,
                         Random &r, int verbosity=0)
{
  vector<UInt> population;
  for (int i= 0; i < sm.nCols(); i++) population.push_back(i);

  vector<UInt> activeBits;
  activeBits.resize(w);

  if (verbosity>0)
  {
    std::cout << "Creating " << M << " random vectors with " << w
            << " bits on.\n";
  }
  for (Int m=0; m < M; m++)
  {
    // Randomly sample from columns
    sample(population.begin(), sm.nCols(), activeBits.begin(), w, r);
    sm.addRow(activeBits.size(), activeBits.begin());

    if (verbosity > 1)
    {
      std::cout << m << ":";
      for (Int i=0; i < w; i++)
      {
        std::cout << activeBits[i] << " ";
      }
      std::cout << std::endl;
    }
  }

}

// Do a single classification trial. Given values for n, w, and M create N
// random vectors plus a random trial vector. For each value of theta from 1 to
// w, return the number of vectors that match.
void classificationFalseMatchTrial(
          UInt n, UInt w, UInt M,
          vector<UInt> &matchesWithThetas, Random &r)
{
  UInt32 population[n];
  for (Int i=0; i < n; i++) population[i] = i;

  // Generate our set of random sparse vectors and store in our "classifier"
  SparseMatrix01<UInt, Int> classifier(n, 1);
  createRandomVectors(M, w, classifier, r);

  // Generate our single random vector
  vector<UInt> x;
  x.resize(w, 0);
  sample(population, n, x.begin(), w, r);

  // Generate number of matches for each value of theta
  for (UInt theta = 1; theta <= w; theta++)
  {
    matchesWithThetas[theta] = numMatches(classifier, x, theta);
    //std::cout << "theta= " << theta << ", num matches= "
    //          << matchesWithThetas[theta] << "\n";
  }

}

// Given values for n, w, M, compute the probability of a false match for
// each value of theta = [1,w]. This is done by performing nTrials separate
// simulations, and seeing how often there is at least one match.
void classificationFalseMatchProbability(UInt n, UInt w, UInt M,
               vector<Real> &probWithThetas, UInt nTrials,
               Random &r)
{
  probWithThetas.clear();
  probWithThetas.resize(w+1, 0.0);

  for (int trial = 0; trial < nTrials; trial++)
  {
    vector<UInt> matchesWithThetas;
    matchesWithThetas.resize(w+1, 0);
    classificationFalseMatchTrial(n, w, M, matchesWithThetas, r);
    if (trial%50000 == 0)
    {
      std::cout << trial << " trials completed out of " << nTrials << "\n";
    }

    for (UInt theta = 1; theta <= w; theta++)
    {
      if (matchesWithThetas[theta] > 0)
      {
        probWithThetas[theta]++;
      }
    }
  }

  std::cout << "Classification: Probability of false match for n=" << n
            << ", M=" << M << ", w=" << w << "\n";
  for (UInt theta = 1; theta <= w; theta++)
  {
    probWithThetas[theta] = (Real) probWithThetas[theta] / (Real) nTrials;
    std::cout << "    Theta = " << theta
              << " prob=" << probWithThetas[theta] << "\n";
  }
}

// Run the trials!  Currently need to hard code the specific trial you are
// about to run.
int main(int argc, char * argv[]) {

  Random r;
  vector<Real> probWithThetas;

  std::cout << "Simulations running. Please be patient. Think about\n"
            << "all the things you have to be grateful for.\n\n";

  classificationFalseMatchProbability(100, 7, 100, probWithThetas, 500000, r);
}

