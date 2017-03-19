// Numenta Platform for Intelligent Computing (NuPIC)
// Copyright (C) 2017, Numenta, Inc.  Unless you have an agreement
// with Numenta, Inc., for a separate license for this software code, the
// following terms and conditions apply:
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero Public License version 3 as
// published by the Free Software Foundation.
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Affero Public License for more details.
// You should have received a copy of the GNU Affero Public License
// along with this program.  If not, see http://www.gnu.org/licenses.
// http://numenta.org/licenses/

using UnityEngine;
using Numenta.Utils;
using System;

namespace Numenta.Nupic
{
    /// <summary>
    /// Wraps NIK HTM python script
    /// </summary>
    public class NikHtm : Python
    {
        public NikHtm() : base("nik_htm.py")
        {
            // Empty
        }

        /// <summary>
        /// Load  TM from the given filename
        /// </summary>
        /// <param name="filename"></param>
        public void Load(string filename)
        {
            WriteLine("load," + filename);
        }

        /// <summary>
        /// Save TM in the given filename
        /// </summary>
        /// <param name="filename"></param>
        public void Save(string filename)
        {
            WriteLine("save," + filename);
        }

        /// <summary>
        /// Train Nik HTM model
        /// </summary>
        /// <param name="x_0">Previous X position</param>
        /// <param name="y_0">Previous Y position</param>
        /// <param name="x_1">Current X position</param>
        /// <param name="y_1">Current Y position</param>
        /// <param name="theta1_0">Previous theta1 (shoulder) angle</param>
        /// <param name="theta2_0">Previous theta2 (elbow) angle</param>
        /// <param name="theta1_1">Current theta1 (shoulder) angle</param>
        /// <param name="theta2_1">Current theta2 (elbow) angle</param>
        public void Train(float x_0, float y_0,
                          float x_1, float y_1,
                          int theta1_0, int theta2_0,
                          int theta1_1, int theta2_1)
        {
            WriteLine(string.Format("{0},{1},{2},{3},{4},{5},{6},{7},true",
                x_0, y_0, x_1, y_1, theta1_0, theta2_0, theta1_1, theta2_1
            ));
        }

        /// <summary>
        /// Infer Arm pose using Nik HTM
        /// </summary>
        /// <param name="x_0">Previous X position</param>
        /// <param name="y_0">Previous Y position</param>
        /// <param name="x_1">Current X position</param>
        /// <param name="y_1">Current Y position</param>
        /// <param name="theta1_0">Previous theta1 (shoulder) angle</param>
        /// <param name="theta2_0">Previous theta2 (elbow) angle</param>
        /// <returns>
        /// Next Pose (theta1 and theta2)
        /// </returns>
        public int[] Infer(float x_0, float y_0,
                             float x_1, float y_1,
                             int theta1_0, int theta2_0)
        {
            WriteLine(string.Format("{0},{1},{2},{3},{4},{5},{6},{7},false",
                x_0, y_0, x_1, y_1, theta1_0, theta2_0, theta1_0, theta2_0
            ));
            Debug.Log(string.Format("{0},{1},{2},{3},{4},{5},{6},{7},false",
                x_0, y_0, x_1, y_1, theta1_0, theta2_0, theta1_0, theta2_0
            ));
            // Read predicted angles
            var line = ReadLine();
            Debug.Log("Result:" + line);
            return Array.ConvertAll(line
                   .Trim(new char[] { '(', ')' })
                   .Split(new char[] { ',' }, StringSplitOptions.RemoveEmptyEntries),
                   x => (int)Convert.ToDouble(x));
        }
    }
}