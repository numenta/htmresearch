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

#include "sdr_utilities.hpp"

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
  x.resize(w, 0);

  UInt matches = 0;
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    matchesWithThetas[theta] = 0;
  }

  for (UInt i = 0; i < k; i++)
  {
    sample(population, n, x.begin(), w, r);

    //vector<UInt> y;
    //y.resize(w, 0);
    //classifier.getRowSparse(44, y.begin());
    //cout << "stored 44: ";
    //for (UInt i = 0; i < w; i++)
    //{
    //  cout << y[i] << " ";
    //}
    //cout << endl;
    matches = numMatches(classifier, x, 1);

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
    if (verbosity > 1 && trial % 100 == 0)
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
      cout << "    Theta = " << theta << " prob=" << probWithThetas[theta]
           << " +/- " << bounds
           << endl;
    }
  }
}

// Do a single classification trial. Given values for n, w, and M create M
// random vectors plus k random trial vectors. For each value of theta from 1 to
// w, return the number of trial vectors that had a match.
void unionClassificationFalseMatchTrial(
    UInt n, UInt w, UInt w_p, UInt M, UInt k, vector<UInt> &matchesWithThetas,
    Random &r)
{
  NTA_ASSERT(w_p <= w < n);

  UInt32 population[n];
  for (Int i=0; i < n; i++) population[i] = i;

  set<UInt> classifier;
  unionRandomVectors(M, n, w, classifier, r);

  // Generate our single random vector
  vector<UInt> x;
  x.resize(w, 0);

  UInt matches = 0;
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    matchesWithThetas[theta] = 0;
  }

  for (UInt i = 0; i < k; i++)
  {
    sample(population, n, x.begin(), w, r);

    //vector<UInt> y;
    //y.resize(w, 0);
    //classifier.getRowSparse(44, y.begin());
    //cout << "stored 44: ";
    //for (UInt i = 0; i < w; i++)
    //{
    //  cout << y[i] << " ";
    //}
    //cout << endl;
    //matches = unionNumMatches(classifier, x, 1);

    // Generate number of matches for each value of theta
    for (UInt theta = 1; theta <= w_p; theta++)
    {
      matches = unionNumMatches(classifier, x, theta);
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
void unionClassificationFalseMatchProbability(
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
    unionClassificationFalseMatchTrial(n, w, w_p, M, k, matchesWithThetas, r);
    if (verbosity > 1 && trial % 100 == 0)
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
      cout << "    Theta = " << theta << " prob=" << probWithThetas[theta]
           << " +/- " << bounds
           << endl;
    }
  }
}

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
    if (trial % 10 == 0)
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

void runOneTrial(UInt n, UInt w, UInt w_p, UInt M, UInt k,
                 vector<Real>& probWithThetas, UInt nTrials, Random& r,
                 ofstream& f, UInt verbosity)
{
  classificationFalseMatchProbability(n, w, w_p, M, k, probWithThetas,
                                      nTrials, r, verbosity);
  f << n << ","
    << w << ","
    << w_p << ","
    << M << ","
    << k << ","
    << nTrials;
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    f << "," << probWithThetas[theta];
  }
  f << endl;
}

void unionRunOneTrial(UInt n, UInt w, UInt w_p, UInt M, UInt k,
                 vector<Real>& probWithThetas, UInt nTrials, Random& r,
                 ofstream& f, UInt verbosity)
{
  cout << "Classification with unions of SDRs:\n";
  f    << "Classification with unions of SDRs:\n";
  f    << "n,w,w_p,M,k,nTrials,Prob with various thetas\n";

  unionClassificationFalseMatchProbability(n, w, w_p, M, k, probWithThetas,
                                           nTrials, r, verbosity);
  f << n << ","
    << w << ","
    << w_p << ","
    << M << ","
    << k << ","
    << nTrials;
  for (UInt theta = 1; theta <= w_p; theta++)
  {
    f << "," << probWithThetas[theta];
  }
  f << endl;
}

void runTrialRange(
    UInt n1, UInt n2, UInt w1, UInt w2, UInt M, UInt k,
    UInt nTrials, Random &r, ofstream& f,
    UInt verbosity)
{
  vector<Real> probWithThetas;

  UInt nDiff = (n2 - n1) / 4;
  for (UInt n = n1; n <= n2; n += nDiff)
  {
    UInt wDiff = (w2 - w1) / 2;
    for (UInt w = w1; w <= w2; w += wDiff)
    {
      UInt wps = w / 3;
      UInt wpe = w * 2 / 3;
      UInt wpdiff = (wpe - wps) / 2;
      for (UInt w_p = wps; w_p < wpe; w_p += wpdiff)
      {
        runOneTrial(n, w, w_p, M, k, probWithThetas, nTrials, r, f, verbosity);
      }
    }
  }
}

// Run the trials!  Currently need to hard code the specific trial you are
// about to run.
int main(int argc, char * argv[])
{
  if (argc < 2)
  {
    cout << "Missing output path argument" << endl;
    exit(1);
  }
  const char* outputPath = argv[1];

//  // This is for false negatives, commented out
//  /*
//  // noise (False negative only)
//  UInt noise = 5;
//  // False negative
//  classificationFalseNegativeProbability(n, w, w_p, M, k, noise, probWithThetas,
//                                         trials, r);
//  */
//
//  // verbosity
//  Byte verbosity = 1;
//  // random number generator
//  Random r(42);
//  // output values where index is theta and value is probability
//  vector<Real> probWithThetas;
//
//  if (verbosity > 0)
//  {
//    cout << "Simulations running. Please be patient. Think about\n"
//         << "all the things you have to be grateful for.\n\n";
//  }
//
//  //// TODO: Set float precision to max.
  ofstream f(outputPath);
  //runTrialRange(1000, 50000, 30, 180, 1, 100, 10000, r, f, verbosity);
  ////runOneTrial(n, w, w_p, M, k, probWithThetas, nTrials, r, f, verbosity);
  //runOneTrial(1000, 20, 20, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(1000, 40, 40, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(1000, 60, 60, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(1000, 80, 80, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(1000, 100, 100, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(5000, 20, 20, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(5000, 40, 40, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(5000, 60, 60, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(5000, 80, 80, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(5000, 100, 100, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(10000, 20, 20, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(10000, 40, 40, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(10000, 60, 60, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(10000, 80, 80, 1, 100, probWithThetas, 10000, r, f, verbosity);
  //runOneTrial(10000, 100, 100, 1, 100, probWithThetas, 10000, r, f, verbosity);

  //runOneTrial(n, w, w_p, M, k, probWithThetas, nTrials, r, f, verbosity);
  //unionRunOneTrial(1000, 20, 20, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(1000, 40, 40, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(1000, 60, 60, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(1000, 80, 80, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(1000, 100, 100, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(5000, 20, 20, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(5000, 40, 40, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(5000, 60, 60, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(5000, 80, 80, 20, 100, probWithThetas, 10000, r, f, verbosity);
  //unionRunOneTrial(5000, 100, 100, 20, 100, probWithThetas, 10000, r, f, verbosity);
//  unionRunOneTrial(10000, 20, 20, 20, 100, probWithThetas, 10000, r, f, verbosity);
//  unionRunOneTrial(10000, 40, 40, 20, 100, probWithThetas, 10000, r, f, verbosity);
//  unionRunOneTrial(10000, 60, 60, 20, 100, probWithThetas, 10000, r, f, verbosity);
//  unionRunOneTrial(10000, 80, 80, 20, 100, probWithThetas, 10000, r, f, verbosity);
//  unionRunOneTrial(10000, 100, 100, 20, 100, probWithThetas, 10000, r, f, verbosity);
  f.close();
}

