/* ---------------------------------------------------------------------
 * Numenta Platform for Intelligent Computing (NuPIC)
 * Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
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

#ifndef BINARY_ALGORITHMS_HPP
#define BINARY_ALGORITHMS_HPP

#include <cassert>
#include <iostream>

#include <nupic/types/Types.hpp>

namespace nupic
{
  const UInt64 ZERO = 0;
  const UInt64 ONE = 1;

  void printDense(UInt64* sdr, UInt n)
  {
    UInt64 one = 1;
    UInt blockIndex = 0;
    UInt64 bitIndex = 0;
    for (UInt i = 0; i < n; i++)
    {
      blockIndex = i / 64;
      bitIndex = i % 64;
      if (sdr[blockIndex] & (one << bitIndex))
      {
        std::cout << "1";
      } else {
        std::cout << "0";
      }
    }
    std::cout << std::endl;
  }

  void printDenseIndices(UInt64* sdr, UInt n)
  {
    UInt64 one = 1;
    UInt blockIndex = 0;
    UInt64 bitIndex = 0;
    for (UInt i = 0; i < n; i++)
    {
      blockIndex = i / 64;
      bitIndex = i % 64;
      if (sdr[blockIndex] & (one << bitIndex))
      {
        std::cout << i << " ";
      }
    }
    std::cout << std::endl;
  }

  template <typename ValueType>
  void printSparseIndices(ValueType &indices, UInt w)
  {
    for (UInt i = 0; i < w; i++)
    {
      std::cout << indices[i] << " ";
    }
    std::cout << std::endl;
  }

  void sparseToDense(UInt64* activeBits, UInt64 w, UInt64* sdr, UInt64 n)
  {
    UInt nBlocks = ((n-1)/64) + 1;
    // Zero the output SDR
    for (UInt i = 0; i < nBlocks; i++) sdr[i] = ZERO;

    UInt blockIndex = 0;
    UInt64 bitIndex = 0;
    for (UInt i = 0; i < w; i++)
    {
      assert (activeBits[i] < n);
      blockIndex = activeBits[i] / 64;
      bitIndex = activeBits[i] % 64;
      // Set the bit at bitIndex in the appropriate sdr block to 1
      sdr[blockIndex] = sdr[blockIndex] | (ONE << bitIndex);
    }
  }

  // n is the number of bits in each representation
  int overlap(UInt64* a, UInt64* b, UInt64 n)
  {
    int count = 0;
    UInt nBlocks = ((n-1)/64) + 1;
    UInt64 combined;
    for (UInt i = 0; i < nBlocks; i++)
    {
      combined = a[i] & b[i];

      // Take 1: Count the on bits, incrementing count
      //while (combined)
      //{
      //  combined &= (combined-1);
      //  count++;
      //}
      // Take 2: Builtin population count
      //count += __builtin_popcountll(combined);
      // Take 3: "Best" method with 64 bit ints
      typedef UInt64 T;
      combined = combined - ((combined >> 1) & (T)~(T)0/3);
      combined = (combined & (T)~(T)0/15*3) + ((combined >> 2) & (T)~(T)0/15*3);
      combined = (combined + (combined >> 4)) & (T)~(T)0/255*15;
      count += (T)(combined * ((T)~(T)0/255)) >> (sizeof(T) - 1) * CHAR_BIT;
    }
    return count;
  }

}  // namespace nupic

#endif // BINARY_ALGORITHMS_HPP
