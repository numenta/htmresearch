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
#include <string>
#include <vector>
#include <utility>

#include <nupic/math/Math.hpp>
#include <nupic/math/SparseMatrix.hpp>
#include <nupic/math/SparseMatrix01.hpp>
#include <nupic/utils/Random.hpp>

#include "binary_algorithms.hpp"


using namespace std;
using namespace nupic;

#include "sdr_utilities.hpp"


// Create M different bit arrays, each with w random 1 bits, rest 0s.
// The representations will use binary long[] with at least n bits. Any
// trailing bits will be zeros and are not part of the representation.
void createRandomSDRs(UInt M, UInt n, UInt w, Random &r, BinaryMatrix& classifier,
                      int verbosity=0)
{
  UInt64 population[n];
  for (UInt64 i= 0; i < n; i++) population[i] = i;

  UInt64 activeBits[w];

  if (verbosity > 0)
  {
    cout << "Creating " << M << " random vectors with " << w << " bits on.\n";
  }
  for (UInt m=0; m < M; m++)
  {
    // Randomly sample bits for each SDR and add the binary representation to
    // the output sdrs array
    sample(population, n, activeBits, w, r);
    classifier.setRowSparse(m, activeBits, w);

    if (verbosity > 1)
    {
      cout << m << ":";
      for (UInt i = 0; i < w; i++)
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
  for (UInt i=0; i < n; i++) population[i] = i;

  // Create a list of stored patterns to put in the classifier
  BinaryMatrix classifier(M, n);
  createRandomSDRs(M, n, w_p, r, classifier);

  // x will hold the w non-zero indices for our single random vector
  UInt64* x = new UInt64[w];

  UInt matches = 0;
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    matchesWithThetas[theta] = 0;
  }

  for (UInt i = 0; i < k; i++)
  {
    sample(population, n, x, w, r);

//    cout << "Our random vector x:\n";
//    printSparseIndices(x, w);

    //UInt64* y = new UInt64[w];
    //classifier.getRowSparse(44, y);
    //cout << "stored 44: ";
    //for (UInt i = 0; i < w; i++)
    //{
    //  cout << y[i] << " ";
    //}
    //cout << endl;
    matches = classifier.matchSparse(x, w, 1);

    // Generate number of matches for each value of theta
    for (UInt theta = 1; theta <= w_p; theta++)
    {
      matches = classifier.matchSparse(x, w, theta);
      if (matches > 0)
      {
        matchesWithThetas[theta]++;
      }
      //cout << "theta= " << theta << ", num matches= "
      //          << matchesWithThetas[theta] << "\n";
    }
  }

  delete[] x;
}

// Given values for n, w, w_p, M, compute the probability of a false match for
// each value of theta = [1,w_p]. This is done by performing nTrials separate
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
    UInt nTrials, Random &r, Byte verbosity)
{
  NTA_ASSERT(w_p <= w < n);

  probWithThetas.clear();
  probWithThetas.resize(w_p+1, 0.0);

  for (int trial = 0; trial < nTrials; trial++)
  {
    vector<UInt> matchesWithThetas;
    matchesWithThetas.resize(w_p+1, 0);
    classificationFalseMatchTrial(n, w, w_p, M, k, matchesWithThetas, r);
    if (verbosity > 0 && (trial>0) && trial % 20000 == 0)
    {
      cout << trial << " trials completed out of " << nTrials << "\n";
    }

    for (UInt theta = 1; theta <= w_p; theta++)
    {
      probWithThetas[theta] += matchesWithThetas[theta];
    }
  }

  if (verbosity > 0)
  {
    cout << "Classification: Probability of false match for n=" << n
         << ", M=" << M << ", w=" << w << ", w'=" << w_p << "\n";
  }
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    probWithThetas[theta] = (Real) probWithThetas[theta] / (Real)(nTrials*k);
    if (verbosity > 0)
    {
      auto bounds = estimateBounds(probWithThetas[theta], nTrials*k);
      if (theta==12)
      {
        cout << "    Theta = " << theta << " prob=" << probWithThetas[theta]
             << " +/- " << bounds
             << endl;
      }
    }
  }
}

/*
// Do a single classification trial. Given values for n, w, and M create N
// random vectors plus a random trial vector. For each value of theta from 1 to
// w, return the number of vectors that match.
void classificationFalseNegativeTrial(
    UInt n, UInt w, UInt w_p, UInt M, UInt k, UInt noise,
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
  UInt matches = 0;
  UInt32 ri;
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    matchesWithThetas[theta] = 0;
  }

  NTA_ASSERT(storedPatterns.nRows() == M);

  for (UInt i = 0; i < k; i++)
  {
    ri = r.getUInt32(storedPatterns.nRows());
    storedPatterns.getRowSparse(ri, x.begin());

    vector<UInt> xp;
    addNoise(x, xp, n, w, noise, r);

    // Generate number of matches for each value of theta
    for (UInt theta = 1; theta <= w_p; theta++)
    {
      matches = numMatches(classifier, xp, theta);
      if (matches == 0)
      {
        matchesWithThetas[theta]++;
      }
      //cout << "theta= " << theta << ", num matches= "
      //          << matchesWithThetas[theta] << "\n";
    }
  }
}

// Given values for n, w, M, compute the probability of a false match for
// each value of theta = [1,w]. This is done by performing nTrials separate
// simulations, and seeing how often there is at least one match.
void classificationFalseNegativeProbability(
    UInt n, UInt w, UInt w_p, UInt M, UInt k, UInt noise,
    vector<Real> &probWithThetas, UInt nTrials, Random &r)
{
  NTA_ASSERT(w_p <= w < n);

  probWithThetas.clear();
  probWithThetas.resize(w_p+1, 0.0);

  for (int trial = 0; trial < nTrials; trial++)
  {
    vector<UInt> matchesWithThetas;
    matchesWithThetas.clear();
    matchesWithThetas.resize(w_p+1, 0);

    classificationFalseNegativeTrial(n, w, w_p, M, k, noise,
                                     matchesWithThetas, r);
    if (trial % 1000 == 0)
    {
      cout << trial << " trials completed out of " << nTrials << "\n";
    }

    for (UInt theta = 1; theta <= w_p; theta++)
    {
      if (matchesWithThetas[theta] > 0)
      {
        probWithThetas[theta] += matchesWithThetas[theta];
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
*/

// Run the trials!  Currently need to hard code the specific trial you are
// about to run.
// The following command line arguments are expected, in this particular order
//   FILE NUM_TRIALS n w
// For example:
//   ./sdr_calculations2 stdout 100 500 64
int main(int argc, char * argv[]) {
  if (argc != 5)
  {
    cout << "Wrong number of arguments!" << endl;
    cout << "The following command line arguments are expected, in this particular order\n"
         << "  FILE NUM_TRIALS n w\n"
         << "For example:\n"
         << "  " << argv[0] << " stdout 100 500 64\n";
    exit(1);
  }

  string outPath(argv[1]);

  // number of trials
  UInt trials = atoi(argv[2]);
  // number of total bits in each representation
  UInt n = atoi(argv[3]);
  // number of active bits in each representation
  UInt w = atoi(argv[4]);
  // w', number of bits to subsample and store for each representation
  UInt w_p = 24;
  // number of patterns to generate and store
  UInt M = 1;
  // number of patterns to test for each trial rather than doing just a
  // single sample per trial - this is purely to speed things up
  UInt k = 100;
  // verbosity
  Byte verbosity = 1;
  // random number generator
  Random r(42);
  // output values where index is theta and value is probability
  vector<Real> probWithThetas;

  // noise (False negative only)
  //UInt noise = 5;

  if (verbosity > 0)
  {
    cout << "Simulations running. Please be patient. Think about\n"
         << "all the things you have to be grateful for.\n\n";
  }

  if (true)
  {
    // False positive
    classificationFalseMatchProbability(n, w, w_p, M, k, probWithThetas,
                                        trials, r, verbosity);
  } else {
    // False negative
    //classificationFalseNegativeProbability(n, w, w_p, M, k, noise, probWithThetas,
    //                                       trials, r);
  }

  // TODO: Set float precision to max.
  ofstream f(outPath, std::ofstream::app);
  for (UInt theta = 12; theta <= 12; theta++)
  {
    f << theta << "," << n << "," << w << "," << probWithThetas[theta] << endl;
  }
  f.close();
}

