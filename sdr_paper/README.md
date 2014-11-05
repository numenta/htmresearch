
This directory contains code and Excel files that simulate various scenarios in 
the SDR paper. They are used to generate the numerical results. The code also
serves to verify the formulas in the paper.

sdr_calculations.cpp
====================

This C++ program simulates SDR classification with various parameters.  In 
particular it simulates the "Classifying a Set of Vectors" setup. You can set
values for M, n, and w and it will return the probability of a false match 
for various values of theta. It computes these probabilities via simulations 
(i.e. generating millions of random vectors and using a classifier to compute
whether there was a match or not). 

As such this can be used to verify that the math in the paper is accurate. 

However, we can only really verify the math for relatively small parameters 
this way. Many of the probabilities are insanely low and it is impossible to 
verify via simulation.  Still if the math matches the simulation results 
exactly for a wide range of parameters, we can be reasonably sure it is 
accurate.

Currently you need to hard code the numbers in the C++ file,  compile and run.
Sorry about that. Here is an example run on my laptop:

For n=100, M=500, w=7 and setting nTrials to 1,000,000 it takes 10 minutes to
generate the following result:
```
Classification: Probability of false match for n=100, M=500, w=7
    Theta = 1 prob=1
    Theta = 2 prob=1
    Theta = 3 prob=0.964839
    Theta = 4 prob=0.134632
    Theta = 5 prob=0.00285
    Theta = 6 prob=1.7e-05
    Theta = 7 prob=0
```

These correspond quite closely to the formula in the paper. For example, for
theta=5 the formula leads to 0.002826477. For theta=4 the formula leads to
0.144691001.  For theta=6 the formula leads to 2.03654E-05  

Remember  that  the formula is an upper bound on the classification error. It 
will be less accurate as you get closer to 1.0. The simulations themselves will
be less accurate for really tiny numbers since we are running a finite number of
trials.  As an example, For theta=7 the  formula leads to 3.12352E-08 but we
need to run hundreds of millions of  trials to get that number through
simulations.

SDR Properties Calculations.xlsx
================================

This excel file contains the formulas in the SDR papers. You can plug in 
different numbers and it tells you the result of various formulas.  Excel  uses
very high precision math, so you can compute the numbers for any  reasonable set
of parameters. (A much wider range than you can by simulations.) This
spreadsheet was used to compute the numbers in the examples and the tables in
the SDR paper. 


Note on using NuPIC Core
========================

The executable `sdr_calculations` is a C++ program that is a client of 
`nupic.core`. As such it is an example of how to write C++ programs that use
that repository as an external library. I have tried to make it as simple as
possible. 

There is a single, very simplistic, Makefile that shows how to compile and link
such code.  The Makefile assumes the environment variable `NUPIC_CORE` is
already setup to point to the root of your `nupic.core` repository.  It 
assumes you have followed the `nupic.core` build instructions and have a 
proper build in place. 

