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

  extern void printDense(UInt64* sdr, UInt n);

  extern void printDenseIndices(UInt64* sdr, UInt n);

  template <typename ValueType>
  void printSparseIndices(ValueType &indices, UInt w)
  {
    for (UInt i = 0; i < w; i++)
    {
      std::cout << indices[i] << " ";
    }
    std::cout << std::endl;
  }

  extern void sparseToDense(UInt64* activeBits, UInt64 w, UInt64* sdr, UInt64 n);

  // n is the number of bits in each representation
  extern int overlap(UInt64* a, UInt64* b, UInt64 n);

  class BinaryMatrix
  {
    public:

      BinaryMatrix(UInt64 nRows, UInt64 nCols)
      {
        nRows_ = nRows;
        nCols_ = nCols;
        data_ = new UInt64*[nRows];
        for (UInt i = 0; i < nRows; i++)
        {
          data_[i] = new UInt64[nCols];
        }
        tempSDR_ = new UInt64[nCols];
      }

      virtual ~BinaryMatrix()
      {
        delete[] tempSDR_;
        for (UInt i = 0; i < nRows_; i++)
        {
          delete[] data_[i];
        }
        delete[] data_;
      }

      UInt64* getRow(UInt64 i)
      {
        return data_[i];
      }

      // Set the specified bits in sdr to 1, rest 0. The bits array
      // is length w. The sdr long array is length ceil(n/64)
      void setRowSparse(UInt64 ri, UInt64* activeBits, UInt64 w)
      {
        sparseToDense(activeBits, w, data_[ri], nCols_);
        //UInt nBlocks = ((nCols_-1)/64) + 1;
        //UInt64* row = data_[ri];
        //// Zero the output SDR
        //for (UInt i = 0; i < nBlocks; i++) row[i] = ZERO;

        //UInt blockIndex = 0;
        //UInt64 bitIndex = 0;
        //for (UInt i = 0; i < w; i++)
        //{
        //  assert (activeBits[i] < nCols_);
        //  blockIndex = activeBits[i] / 64;
        //  bitIndex = activeBits[i] % 64;
        //  // Set the bit at bitIndex in the appropriate sdr block to 1
        //  row[blockIndex] = row[blockIndex] | (ONE << bitIndex);
        //}
      }

      void getRowSparse(UInt64 ri, UInt64* output)
      {
        UInt blockIndex = 0;
        UInt64 bitIndex = 0;
        UInt outIndex = 0;
        for (UInt i = 0; i < nCols_; i++)
        {
          blockIndex = i / 64;
          bitIndex = i % 64;
          if (data_[ri][blockIndex] & (ONE << bitIndex))
          {
            output[outIndex++] = i;
          }
        }
      }

      UInt64 matchSparse(UInt64* indices, UInt64 w, UInt theta)
      {
        sparseToDense(indices, w, tempSDR_, nCols_);
        return matchDense(tempSDR_, theta);
      }

      UInt64 matchDense(UInt64* sdr, UInt theta)
      {
        UInt64 matches = 0;
        for (UInt i = 0; i < nRows_; i++)
        {
          if (overlap(data_[i], sdr, nCols_) >= theta)
          {
            matches++;
          }
        }
        return matches;
      }

      void print()
      {
        for (UInt i = 0; i < nRows_; i++)
        {
          printDense(data_[i], nCols_);
        }
      }

    private:
      UInt64 nRows_;
      UInt64 nCols_;
      UInt64** data_;
      UInt64* tempSDR_;
  };

}  // namespace nupic

#endif // BINARY_ALGORITHMS_HPP
