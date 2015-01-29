/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2014-2015, Numenta, Inc.  Unless you have an agreement
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

#include <cassert>
#include <iostream>

#include <nupic/types/Types.hpp>

#include "binary_algorithms.hpp"

using namespace nupic;
using namespace std;

int main(int argc, char * argv[])
{
  {
    BinaryMatrix classifier(3, 66);

    UInt64 indices1[4] = {0, 1, 2, 3};
    classifier.setRowSparse(0, indices1, 4);
    UInt64 indices2[4] = {1, 3, 63, 64};
    classifier.setRowSparse(1, indices2, 4);
    UInt64 indices3[4] = {3, 63, 64, 65};
    classifier.setRowSparse(2, indices3, 4);

    classifier.print();

    const UInt64 ONE = 1;
    UInt64 sdr[2] = {8, 2};
    UInt64 matches = classifier.matchDense(sdr, 2);
    cout << "matches: " << matches << endl;
  }
}
