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

using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Numenta.Utils;
using Numenta.Controllers;
using Numenta.Nupic;

/// <summary>
/// Numenta Inverse Kinematics (Nik)
///
/// The following keys control the experiment:
///  - 'T' : Start training
///  - '0' : Infer new pose from current state
///  - '1', '2', '3': Set experiment to predetermined poses 1, 2 or 3
///  - 'D' : Print debug information
/// </summary>
namespace Numenta
{
    public class Experiment1 : MonoBehaviour
    {
        [Tooltip("Shoulder angle step resolutions in degrees")]
        public int shoulderResolution = 5;
        [Tooltip("Elbow angle step resolutions in degrees")]
        public int elbowResolution = 15;
        [Tooltip("Total random points to use during training")]
        public int trainingSize = 10;
        [Tooltip("Whether or not to save the training data log")]
        public bool saveLog = true;
        [Tooltip("Whether or not to Nik for inverse kinematics")]
        public bool useNik = false;
        [Tooltip("Pre-trainded serialized model name to use. Path is relative to the 'Assets' folder")]
        public string serializedModelName;
        [Tooltip("Object to seek using inverse kinematics")]
        public Transform target;
        [Tooltip("Random number generator seed")]
        public int seed = 37;
        public GUIStyle style;

        List<string> log;
        MersenneTwister rng;
        string logDataPath;
        ArmController arm;
        NikHtm nik;
        bool isTraining = false;
        bool loading = false;

        void Start()
        {
            rng = new MersenneTwister(seed);
            arm = GetComponent<ArmController>();

            // Logging
            log = new List<string>();
            logDataPath = Path.Combine(Application.persistentDataPath, "data");
            logDataPath = Path.Combine(logDataPath, name);
            if (!Directory.Exists(logDataPath))
            {
                Directory.CreateDirectory(logDataPath);
            }
            // Initialize Nik
            if (useNik)
            {
                nik = new NikHtm();
                nik.Start();
                if (serializedModelName != null && serializedModelName.Trim().Length > 0)
                {
                    StartCoroutine(Load());
                }
            }
            isTraining = false;
        }

        void Update()
        {
            if (Input.GetKey(KeyCode.T))
            {
                StartCoroutine(Train());
            }
            if (Input.GetKey(KeyCode.Alpha0))
            {
                StartCoroutine(Infer(target.position));
            }
            if (Input.GetKeyUp(KeyCode.Alpha1))
            {
                target.position = new Vector3(-2.21f, 3.08f, 0);
                StartCoroutine(arm.Pose(75, 77));
            }
            if (Input.GetKeyUp(KeyCode.Alpha2))
            {
                target.position = new Vector3(-2.21f, 3.45f, 0);
                StartCoroutine(arm.Pose(75, 77));
            }
            if (Input.GetKeyUp(KeyCode.Alpha3))
            {
                target.position = new Vector3(-2.21f, 0, 0);
                StartCoroutine(arm.Pose(75, 133));
            }
            if (Input.GetKeyUp(KeyCode.D))
            {
                Vector2 hand = arm.Hand.position;
                Vector2 balloon = target.position;
                Vector3 delta = (balloon - hand);
                float distance = Vector2.Distance(hand, balloon) * 0.02f;

                Debug.Log(distance.ToString("F5"));
                Debug.Log(delta.ToString("F5"));
                Debug.Log((delta * distance).ToString("F5"));
            }
        }

        void OnDestroy()
        {
            if (useNik)
            {
                nik.Stop();
            }
        }

        int[] GetRandomPose()
        {
            return new int[] {
                (int)Mathf.Ceil(rng.Next(0, 90) / shoulderResolution) * shoulderResolution,
                (int)Mathf.Ceil(rng.Next(30, 300) / elbowResolution) * elbowResolution
            };
        }

        int[] GetCurrentPose()
        {
            return new int[]
            {
                (int)Mathf.Ceil(arm.Shoulder.localRotation.eulerAngles.z / shoulderResolution) * shoulderResolution,
                (int)Mathf.Ceil(arm.Elbow.localRotation.eulerAngles.z / elbowResolution) * elbowResolution
            };
        }

        Vector2 GetCurrentPosition()
        {
            return arm.Hand.position;
        }
        /// <summary>
        /// Load Pre-trainded serialized model name to use. Path is relative to the 'Assets' folder
        /// </summary>
        /// <returns></returns>
        IEnumerator Load()
        {
            loading = true;
            yield return null;
            nik.Load(serializedModelName);
            loading = false;
            yield return null;
        }
        IEnumerator Train()
        {
            if (isTraining)
            {
                yield break;
            }
            isTraining = true;
            if (saveLog)
            {
                log.Clear();
            }

            // Start at (0,0)
            yield return arm.Pose(0, 180);

            // Move until all data is logged
            while (log.Count < trainingSize)
            {
                // Get new pose
                int[] targetPose = GetRandomPose();

                // Move Shoulder
                yield return MoveJoint(arm.Shoulder, targetPose[0], shoulderResolution);

                // Move Elbow
                yield return MoveJoint(arm.Elbow, targetPose[1], elbowResolution);
            }
            if (saveLog)
            {
                log.Insert(0, "x(t-1), y(t-1), x(t), y(t), theta1(t-1), theta2(t-1), theta1(t), theta2(t), training");
                File.WriteAllLines(Path.Combine(logDataPath, "train.csv"), log.ToArray());
                log.Clear();
            }
            yield return arm.Pose(60, 60);
            isTraining = false;
        }

        void OnGUI()
        {
            if (loading)
            {
                GUI.skin.label = style;
                GUILayout.Label("Loading...");
            }
        }
        /// <summary>
        /// Move the joint to the given angle
        /// </summary>
        /// <param name="joint"></param>
        /// <param name="theta"></param>
        /// <param name="resolution"></param>
        /// <returns></returns>
        IEnumerator MoveJoint(Transform joint, int theta, int resolution)
        {
            Quaternion targetRotation = Quaternion.Euler(0, 0, theta);
            int[] curPose = GetCurrentPose();
            int remaining = (int)Quaternion.Angle(joint.localRotation, targetRotation);
            while (remaining > 0 && log.Count < trainingSize)
            {
                // Save pose and position before rotation
                Vector2 prev = GetCurrentPosition();
                int[] prevPose = curPose;
                joint.localRotation = Quaternion.RotateTowards(joint.localRotation, targetRotation, resolution);
                yield return new WaitForEndOfFrame();

                // Get pose values and position after rotation
                Vector2 cur = GetCurrentPosition();
                curPose = GetCurrentPose();
                // Train model
                if (useNik)
                {
                    nik.Train(prev.x, prev.y, cur.x, cur.y,
                              prevPose[0], prevPose[1],
                              curPose[0], curPose[1]);
                }

                // Log data
                if (saveLog)
                {
                    log.Add(string.Format("{0},{1},{2},{3},{4},{5},{6},{7},true",
                                            prev.x, prev.y,
                                            cur.x, cur.y,
                                            prevPose[0], prevPose[1],
                                            curPose[0], curPose[1]));
                }
                // Move next step
                remaining = (int)Quaternion.Angle(joint.localRotation, targetRotation);
            }
            yield return null;
        }
        /// <summary>
        /// Coroutine used to infer the next pose based on the given goal
        /// </summary>
        /// <param name="goal"></param>
        IEnumerator Infer(Vector2 goal)
        {
            float scale = 0.2f;
            if (useNik)
            {
                for (int i = 0; i < 40; i++)
                {
                    Vector2 cur = GetCurrentPosition();
                    int[] curPose = GetCurrentPose();
                    Vector2 pos = Vector2.MoveTowards(cur, goal, Vector2.Distance(cur, goal) * scale);
                    Debug.DrawLine(cur, pos, Color.green, 60f);
                    int[] nextPose = nik.Infer(cur.x, cur.y, pos.x, pos.y, curPose[0], curPose[1]);
                    if ((nextPose[0] == 0 && nextPose[1] == 0) ||
                        (nextPose[0] == curPose[0] && nextPose[1] == curPose[1]))
                    {
                        scale = (float)(0.1 + rng.NextDouble() * 0.4);
                        continue;
                    }
                    yield return arm.Pose(nextPose[0], nextPose[1]);
                    yield return new WaitForEndOfFrame();
                    if (arm.touching && arm.touching.transform.Equals(target))
                    {
                        yield break;
                    }
                }
            }
            else
            {
                for (int i = 0; i < 10; i++)
                {
                    Vector2 cur = GetCurrentPosition();
                    int[] curPose = GetCurrentPose();
                    Vector2 pos = Vector2.MoveTowards(cur, goal, Vector2.Distance(cur, goal) * scale);
                    int[] nextPose = InverseKinematics(pos);
                    if ((nextPose[0] == 0 && nextPose[1] == 0) ||
                        (nextPose[0] == curPose[0] && nextPose[1] == curPose[1]))
                    {
                        scale = (float)(0.1 + rng.NextDouble() * 0.4);
                        continue;
                    }
                    yield return arm.Pose(nextPose[0], nextPose[1]);
                }
            }
        }

        /// <summary>
        /// Traditional Inverse kinematics formula
        /// </summary>
        /// <param name="pos"></param>
        /// <returns></returns>
        int[] InverseKinematics(Vector2 pos)
        {
            var l1 = arm.upperArmLength;
            var l2 = arm.lowerArmLength;
            float cos_t2 = (pos.sqrMagnitude - l1 * l1 - l2 * l2) / (2 * l1 * l2);
            float sin_t2 = Mathf.Sqrt(1 - cos_t2 * cos_t2);
            float t2 = Mathf.Atan2(sin_t2, cos_t2);
            float gamma = Mathf.Atan2(l2 * Mathf.Sin(t2), l1 + l2 * (Mathf.Cos(t2)));
            float t1 = Mathf.Atan2(pos.y, pos.x) - gamma;
            return new int[]
            {
                (int)(t1 * Mathf.Rad2Deg),
                (int)(t2 * Mathf.Rad2Deg)
            };
        }
    }
}