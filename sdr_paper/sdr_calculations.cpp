/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014, Numenta, Inc.  Unless you have an agreement
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
#include <string>
#include <vector>
#include <utility>

#include <nupic/math/Math.hpp>
#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SparseMatrix01.hpp>
#include <nupic/utils/Random.hpp>

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

  int numMatches = 0;
  for (int i = 0; i < overlaps.size(); i++)
  {
    if (overlaps[i] >= theta) numMatches++;
  }

  return numMatches;
}

// Change exactly `noise` bits from x and put the result in xp. The number
// added vs removed is determined randomly with equal chance for each
// combination
void addNoise(const vector<UInt>& x, vector<UInt>& xp, UInt n, UInt w,
              UInt noise, Random& r)
{
  // Create a population that does not include the original bits
  vector<UInt32> addOptions;
  for (Int i=0; i < n; i++) addOptions.push_back(i);
  // Iterate in reverse order so indices in addOptions don't change
  for (Int i=w-1; i >= 0; --i) addOptions.erase(addOptions.begin() + x[i]);

  UInt nAdded = r.getUInt32(noise + 1);
  UInt nRemoved = noise - nAdded;

  xp.resize(w + nAdded - nRemoved);
  sample(x.begin(), w, xp.begin(), w - nRemoved, r);
  NTA_ASSERT(addOptions.size() == (n - w));
  sample(addOptions.begin(), addOptions.size(),
         xp.begin() + w - nRemoved, nAdded, r);
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

// Do a single classification trial. Given values for n, w, and M create M
// random vectors plus k random trial vectors. For each value of theta from 1 to
// w, return the number of trial vectors that had a match.
void classificationFalseMatchTrial(
    UInt n, UInt w, UInt w_p, UInt M, UInt k, vector<UInt> &matchesWithThetas,
    Random &r)
{
  NTA_ASSERT(w_p <= w < n);

  UInt32 population[n];
  for (Int i=0; i < n; i++) population[i] = i;

  // Create a list of stored patterns to put in the classifier
  SparseMatrix01<UInt, Int> classifier(n, 1);
  createRandomVectors(M, w, classifier, r);

  // Generate our single random vector
  vector<UInt> x;
  UInt matches = 0;
  x.resize(w, 0);
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    matchesWithThetas[theta] = 0;
  }

  for (UInt i = 0; i < k; i++)
  {
    sample(population, n, x.begin(), w, r);

    // Generate number of matches for each value of theta
    for (UInt theta = 1; theta <= w_p; theta++)
    {
      matches = numMatches(classifier, x, theta);
      if (matches > 0)
      {
        matchesWithThetas[theta]++;
      }
      //cout << "theta= " << theta << ", num matches= "
      //          << matchesWithThetas[theta] << "\n";
    }
  }
}

// Given values for n, w, w_p, M, compute the probability of a false match for
// each value of theta = [1,w]. This is done by performing nTrials separate
// simulations, and seeing how often there is at least one match.
//
// @param n number of bits per vector
// @param w number of active bits per vector
// @param w_p number of bits to subsample and store for each of M vectors
// @param M number of vectors to generate and store in classifier
// @param k number of vectors to generate and test per trial
// @param probWithThetas probabilities of false match for each theta value
// @param nTrials number of trials to run to compute probability
// @param r a random number generator
void classificationFalseMatchProbability(
    UInt n, UInt w, UInt w_p, UInt M, UInt k, vector<Real> &probWithThetas,
    UInt nTrials, Random &r)
{
  NTA_ASSERT(w_p <= w < n);

  probWithThetas.clear();
  probWithThetas.resize(w_p+1, 0.0);

  for (int trial = 0; trial < nTrials; trial++)
  {
    vector<UInt> matchesWithThetas;
    matchesWithThetas.resize(w_p+1, 0);
    classificationFalseMatchTrial(n, w, w_p, M, k, matchesWithThetas, r);
    if (trial % 10000 == 0)
    {
      cout << trial << " trials completed out of " << nTrials << "\n";
    }

    for (UInt theta = 1; theta <= w_p; theta++)
    {
      probWithThetas[theta] += matchesWithThetas[theta];
    }
  }

  cout << "Classification: Probability of false match for n=" << n
       << ", M=" << M << ", w=" << w << ", w'=" << w_p << "\n";
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    probWithThetas[theta] = (Real) probWithThetas[theta] / (Real)(nTrials*k);
    auto bounds = estimateBounds(probWithThetas[theta], nTrials*k);
    cout << "    Theta = " << theta << " prob=" << probWithThetas[theta]
         << " +/- " << bounds
         << endl;
  }
}

// Do a single classification trial. Given values for n, w, and M create N
// random vectors plus a random trial vector. For each value of theta from 1 to
// w, return the number of vectors that match.
void classificationFalseNegativeTrial(
    UInt n, UInt w, UInt w_p, UInt M, UInt noise,
    vector<UInt> &matchesWithThetas, Random &r)
{
  NTA_ASSERT(noise <= w_p <= w < n);

  UInt32 population[n];
  for (Int i=0; i < n; i++) population[i] = i;

  // Create a list of stored patterns to put in the classifier
  SparseMatrix01<UInt, Int> storedPatterns(n, 1);
  createRandomVectors(M, w, storedPatterns, r);

  // Add subsampled versions of the vectors to the classifier
  SparseMatrix01<UInt, Int> classifier(n, 1);
  vector<UInt> originalRow;
  vector<UInt> subsampledRow;
  for (UInt i=0; i < M; ++i)
  {
    originalRow.clear();
    originalRow.resize(w);
    storedPatterns.getRowSparse(i, originalRow.begin());

    subsampledRow.clear();
    subsampledRow.resize(w_p);
    sample(originalRow.begin(), w, subsampledRow.begin(), w_p, r);
    classifier.addRow(w_p, subsampledRow.begin());
  }

  // Pick one of the stored patterns to test
  vector<UInt> x;
  x.resize(w, 0);
  NTA_ASSERT(storedPatterns.nRows() == M);
  UInt32 ri = r.getUInt32(storedPatterns.nRows());
  storedPatterns.getRowSparse(ri, x.begin());

  vector<UInt> xp;
  addNoise(x, xp, n, w, noise, r);

  // Generate number of matches for each value of theta
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    matchesWithThetas[theta] = numMatches(classifier, xp, theta);
    //cout << "theta= " << theta << ", num matches= "
    //          << matchesWithThetas[theta] << "\n";
  }
}

// Given values for n, w, M, compute the probability of a false match for
// each value of theta = [1,w]. This is done by performing nTrials separate
// simulations, and seeing how often there is at least one match.
void classificationFalseNegativeProbability(
    UInt n, UInt w, UInt w_p, UInt M, UInt noise, vector<Real> &probWithThetas,
    UInt nTrials, Random &r)
{
  NTA_ASSERT(w_p <= w < n);

  probWithThetas.clear();
  probWithThetas.resize(w_p+1, 0.0);

  for (int trial = 0; trial < nTrials; trial++)
  {
    vector<UInt> matchesWithThetas;
    matchesWithThetas.clear();
    matchesWithThetas.resize(w_p+1, 0);

    classificationFalseNegativeTrial(n, w, w_p, M, noise, matchesWithThetas, r);
    if (trial % 10000 == 0)
    {
      cout << trial << " trials completed out of " << nTrials << "\n";
    }

    for (UInt theta = 1; theta <= w_p; theta++)
    {
      if (matchesWithThetas[theta] == 0)
      {
        probWithThetas[theta]++;
      }
    }
  }

  cout << "Classification: Probability of false negative for n=" << n
       << ", M=" << M << ", w=" << w << ", w'=" << w_p << ", noise=" << noise
       << "\n";
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    probWithThetas[theta] = (Real) probWithThetas[theta] / (Real) nTrials;
    auto bounds = estimateBounds(probWithThetas[theta], nTrials);
    cout << "    Theta = " << theta << " prob=" << probWithThetas[theta]
         << " +/- " << bounds
         << endl;
  }
}

// Run the trials!  Currently need to hard code the specific trial you are
// about to run.
int main(int argc, char * argv[]) {

  Random r;
  vector<Real> probWithThetas;

  cout << "Simulations running. Please be patient. Think about\n"
       << "all the things you have to be grateful for.\n\n";

  // False match probability for random vector against 100 stored vectors
  classificationFalseMatchProbability(1024, 30, 30, 100, 1, probWithThetas,
                                      50000, r);
  classificationFalseMatchProbability(1024, 30, 30, 100, 10, probWithThetas,
                                      5000, r);

  // False match probability for random vector against 100 stored vectors where
  // the stored vectors are subsampled (20 out of 30 bits stored)
  //classificationFalseMatchProbability(1024, 30, 20, 100, probWithThetas,
  //                                    50000, r);

  // False negative probability for a stored vector with 5 bits changed against
  // 100 subsampled (20 out of 30 bits stored) random vectors
  //classificationFalseNegativeProbability(1024, 30, 20, 100, 5, probWithThetas,
  //                                       50000, r);
}

