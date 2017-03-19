/*
   C# implementation of Mersenne Twister Random Number Generator adapted from
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/MT2002/emt19937ar.html

   LICENSE:

   A C-program for MT19937, with initialization improved 2002/1/26.
   Coded by Takuji Nishimura and Makoto Matsumoto.

   Before using, initialize the state by using init_genrand(seed)
   or init_by_array(init_key, key_length).

   Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
   All rights reserved.

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

     1. Redistributions of source code must retain the above copyright
        notice, this list of conditions and the following disclaimer.

     2. Redistributions in binary form must reproduce the above copyright
        notice, this list of conditions and the following disclaimer in the
        documentation and/or other materials provided with the distribution.

     3. The names of its contributors may not be used to endorse or promote
        products derived from this software without specific prior written
        permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


   Any feedback is very welcome.
   http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/emt.html
   email: m-mat @ math.sci.hiroshima-u.ac.jp (remove space)
*/

using System;
namespace Numenta.Utils
{
    public class MersenneTwister : System.Random
    {
        /* Period parameters */
        private const int N = 624;
        private const int M = 397;
        private const UInt32 MATRIX_A = 0x9908b0df;   /* constant vector a */
        private const UInt32 UPPER_MASK = 0x80000000; /* most significant w-r bits */
        private const UInt32 LOWER_MASK = 0x7fffffff; /* least significant r bits */
        private UInt32[] mt = new UInt32[N]; /* the array for the state vector  */
        private UInt32 mti = N + 1; /* mti==N+1 means mt[N] is not initialized */
        private UInt32[] mag01 = new UInt32[] { 0, MATRIX_A }; /* mag01[x] = x * MATRIX_A  for x=0,1 */

        public MersenneTwister() : this(Environment.TickCount) { }
        public MersenneTwister(int seed)
        {
            init_genrand((UInt32)seed);
        }
        public MersenneTwister(UInt32[] init_key)
        {
            init_by_array(init_key);
        }
        public MersenneTwister(byte[] buffer)
        {
            if (buffer.Length == 0)
            {
                throw new ArgumentException("buffer is empty", "buffer");
            }
            if (buffer.Length % 4 != 0)
            {
                throw new ArgumentException("buffer size must be multiple of 4", "buffer");
            }
            int size = buffer.Length / 4;
            UInt32[] init_key = new UInt32[size];
            for (int i = 0; i < size; i++)
            {
                init_key[i] = BitConverter.ToUInt32(buffer, i * 4);
            }
            init_by_array(init_key);
        }
        public override int Next()
        {
            return genrand_int31();
        }
        public override int Next(int max)
        {
            int smax = Int32.MaxValue - (Int32.MaxValue % max);
            int sample;
            do
            {
                sample = genrand_int31();
            } while (sample > smax);
            return sample % max;
        }
        public override int Next(int minValue, int maxValue)
        {
            if (minValue > maxValue)
            {
                throw new ArgumentException("minValue must be less than maxValue", "minValue");
            }
            return minValue + Next(maxValue - minValue);
        }
        public override double NextDouble()
        {
            return genrand_real2();
        }
        protected override double Sample()
        {
            return genrand_real2();
        }

        public void SetSeed(int seed)
        {
            init_genrand((UInt32)seed);
        }
        /* initializes mt[N] with a seed */
        void init_genrand(UInt32 s)
        {
            mt[0] = s;
            for (mti = 1; mti < N; mti++)
            {
                mt[mti] = ((UInt32)1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
                /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
                /* In the previous versions, MSBs of the seed affect   */
                /* only MSBs of the array mt[].                        */
                /* 2002/01/09 modified by Makoto Matsumoto             */
            }
        }

        /* initialize by an array with array-length */
        /* init_key is the array for initializing keys */
        /* key_length is its length */
        /* slight change for C++, 2004/2/26 */
        void init_by_array(UInt32[] init_key)
        {
            UInt32 i, j;
            int k;
            int key_length = init_key.Length;

            init_genrand(19650218);
            i = 1; j = 0;
            k = (N > key_length ? N : key_length);

            for (; k > 0; k--)
            {
                mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * (UInt32)1664525))
                    + init_key[j] + (UInt32)j; /* non linear */
                i++; j++;
                if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
                if (j >= key_length) j = 0;
            }
            for (k = N - 1; k > 0; k--)
            {
                mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * (UInt32)1566083941))
                    - (UInt32)i; /* non linear */
                i++;
                if (i >= N) { mt[0] = mt[N - 1]; i = 1; }
            }

            mt[0] = 0x80000000; /* MSB is 1; assuring non-zero initial array */
        }

        /* generates a random number on [0,0xffffffff]-interval */
        UInt32 genrand_int32()
        {
            UInt32 y;

            if (mti >= N)
            { /* generate N words at one time */
                int kk;

                if (mti == N + 1)   /* if init_genrand() has not been called, */
                    init_genrand(5489); /* a default initial seed is used */

                for (kk = 0; kk < N - M; kk++)
                {
                    y = ((mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)) >> 1;
                    mt[kk] = mt[kk + M] ^ mag01[mt[kk + 1] & 1] ^ y;
                }
                for (; kk < N - 1; kk++)
                {
                    y = ((mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK)) >> 1;
                    mt[kk] = mt[kk + (M - N)] ^ mag01[mt[kk + 1] & 1] ^ y;
                }
                y = ((mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK)) >> 1;
                mt[N - 1] = mt[M - 1] ^ mag01[mt[0] & 1] ^ y;

                mti = 0;
            }

            y = mt[mti++];

            /* Tempering */
            y ^= (y >> 11);
            y ^= (y << 7) & 0x9d2c5680;
            y ^= (y << 15) & 0xefc60000;
            y ^= (y >> 18);

            return y;
        }

        /* generates a random number on [0,0x7fffffff]-interval */
        int genrand_int31()
        {
            return (int)(genrand_int32() >> 1);
        }

        /* generates a random number on [0,1]-real-interval */
        double genrand_real1()
        {
            return genrand_int32() * (1.0 / 4294967295.0);
            /* divided by 2^32-1 */
        }

        /* generates a random number on [0,1)-real-interval */
        double genrand_real2()
        {
            return genrand_int32() * (1.0 / 4294967296.0);
            /* divided by 2^32 */
        }

        /* generates a random number on (0,1)-real-interval */
        double genrand_real3()
        {
            return (((double)genrand_int32()) + 0.5) * (1.0 / 4294967296.0);
            /* divided by 2^32 */
        }

        /* generates a random number on [0,1) with 53-bit resolution*/
        double genrand_res53()
        {
            UInt32 a = genrand_int32() >> 5, b = genrand_int32() >> 6;
            return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0);
        }
        /* These real versions are due to Isaku Wada, 2002/01/09 added */
    }
}
