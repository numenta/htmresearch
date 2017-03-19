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
using UnityEngine;

namespace Numenta.Controllers
{
    /// <summary>
    /// Controls Nik Robot Arm
    /// Use "ElbowInput" and ShoulderInput" properties to select Input controller.
    ///
    /// The following keys control the robot behavior:
    ///
    ///  - 'Space' : Random Pose
    ///  - 'R' : Rest Pose (0, 180)
    ///  - 'N' : Initial "Numenta" logo pose (60, 60)
    /// </summary>
    public class ArmController : MonoBehaviour
    {
        /// <summary>
        /// Helper class attached to robot hand used to detect collisions
        /// </summary>
        class Sensor : MonoBehaviour
        {
            public ArmController controller;
            void OnTriggerEnter2D(Collider2D other)
            {
                controller.Touched(other.gameObject);
            }
            void OnTriggerExit2D(Collider2D other)
            {
                controller.Touched(null);
            }
        }

        [Tooltip("Shoulder Joint")]
        public Transform Shoulder;
        [Tooltip("Elbow Joint")]
        public Transform Elbow;
        [Tooltip("Hand Joint")]
        public Transform Hand;
        [Tooltip("Input controller used to move the shoulder joint.\nSee 'Menu > Edit > Project Settings > Input'")]
        public string ShoulderInput = "Horizontal";
        [Tooltip("Input controller used to move the elbow joint.\nSee 'Menu > Edit > Project Settings > Input'")]
        public string ElbowInput = "Vertical";
        [Tooltip("The GameObject this arm's hand is current touching")]
        public GameObject touching;

        float initialShoulderPose;
        float initialElbowPose;
        public float upperArmLength;
        public float lowerArmLength;

        void Start()
        {
            // Save initial pose used by "Restore"
            initialShoulderPose = Shoulder.localRotation.eulerAngles.z;
            initialElbowPose = Elbow.localRotation.eulerAngles.z;

            // Add sensor to Hand
            var handGo = Hand.gameObject;
            var collider = handGo.AddComponent<CircleCollider2D>();
            collider.isTrigger = true;
            var handler = handGo.AddComponent<Sensor>();
            handler.controller = this;

            // Calculate limb length
            upperArmLength = Vector3.Distance(Shoulder.position, Elbow.position);
            lowerArmLength = Vector3.Distance(Elbow.position, Hand.position);
        }

        void Update()
        {
            // Shoulder movements
            var shoulderAngle = Input.GetAxis(ShoulderInput);
            if (shoulderAngle != 0)
            {
                var angle = Shoulder.localRotation.eulerAngles.z + shoulderAngle;
                angle = Mathf.Clamp(angle, 0, 90);
                Shoulder.localRotation = Quaternion.Euler(0, 0, angle);
            }

            // Elbow movements
            var elbowAngle = Input.GetAxis(ElbowInput);
            if (elbowAngle != 0)
            {
                var angle = Elbow.localRotation.eulerAngles.z + elbowAngle;
                angle = angle < 0 ? angle + 360 : angle;
                angle = Mathf.Clamp(angle, 0, 360);
                Elbow.localRotation = Quaternion.Euler(0, 0, angle);
            }

            // Restore initial numenta pose
            if (Input.GetKeyUp(KeyCode.N))
            {
                StopAllCoroutines();
                StartCoroutine(Pose(initialShoulderPose, initialElbowPose));
            }

            // Rest pose
            if (Input.GetKeyUp(KeyCode.R))
            {
                StopAllCoroutines();
                StartCoroutine(Pose(0, 180));
            }

            // Random pose
            if (Input.GetKeyUp(KeyCode.Space))
            {
                StopAllCoroutines();
                StartCoroutine(Pose(Random.Range(0, 90), Random.Range(30, 330)));
            }
        }

        /// <summary>
        /// Message sent whenever the Hand touches other object
        /// </summary>
        /// <param name="other"></param>
        void Touched(GameObject other)
        {
            touching = other;
        }

        /// <summary>
        /// Moves robot arm to the given pose
        /// </summary>
        /// <param name="shoulderAngle"></param>
        /// <param name="elbowAngle"></param>
        /// <returns></returns>
        public IEnumerator Pose(float shoulderAngle, float elbowAngle)
        {
            yield return Rotate(Elbow, Quaternion.Euler(0, 0, elbowAngle));
            yield return Rotate(Shoulder, Quaternion.Euler(0, 0, shoulderAngle));
        }

        /// <summary>
        /// Rotate joint to the given local rotation
        /// </summary>
        IEnumerator Rotate(Transform joint, Quaternion rotation)
        {
            float counter = 0;
            float remaining = Quaternion.Angle(joint.localRotation, rotation);
            while (remaining > 0)
            {
                joint.localRotation = Quaternion.RotateTowards(joint.localRotation, rotation, counter / .1f);
                remaining = Quaternion.Angle(joint.localRotation, rotation);
                counter += Time.deltaTime;
                yield return null;
            }
            yield return null;
        }
    }
}