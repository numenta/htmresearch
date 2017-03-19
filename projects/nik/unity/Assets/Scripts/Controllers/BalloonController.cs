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
    /// Controls the target balloons
    ///
    /// </summary>
    public class BalloonController : MonoBehaviour
    {
        static Quaternion UP = Quaternion.Euler(0, 0, 180);
        static Quaternion DOWN = Quaternion.Euler(0, 0, 0);

        /// <summary>
        /// Helper class used to detect collisions and give some feedback
        /// </summary>
        class Sensor : MonoBehaviour
        {
            public BalloonController controller;
            void OnTriggerEnter2D(Collider2D other)
            {
                float remaining = Quaternion.Angle(controller.balloon.localRotation, UP);
                if (remaining != 0)
                {
                    StopAllCoroutines();
                    StartCoroutine(controller.Rotate(UP));
                }
            }
            void OnTriggerExit2D(Collider2D other)
            {
                float remaining = Quaternion.Angle(controller.balloon.localRotation, DOWN);
                if (remaining != 0)
                {
                    StopAllCoroutines();
                    StartCoroutine(controller.Rotate(DOWN));
                }
            }
        }

        [Tooltip("Balloon Object")]
        public Transform balloon;
        [Tooltip("Random Area position and random.\nUse Transform's scale Z as radius")]
        public Transform randomArea;

        Quaternion initialRotation;
        Vector3 initialLocation;
        CircleCollider2D balloonCollider;

        void Start()
        {
            initialLocation = transform.position;
            initialRotation = balloon.localRotation;
            // Add collider and sensor to balloon
            var ballooonGo = balloon.gameObject;
            balloonCollider = ballooonGo.AddComponent<CircleCollider2D>();
            var handler = ballooonGo.AddComponent<Sensor>();
            handler.controller = this;
            var rb = ballooonGo.AddComponent<Rigidbody2D>();
            rb.isKinematic = true;
        }

        void Update()
        {
            // Restore initial numenta pose
            if (Input.GetKeyUp(KeyCode.N))
            {
                StopAllCoroutines();
                StartCoroutine(Move(initialLocation));
                StartCoroutine(Rotate(initialRotation));
            }
            // Random pose
            if (Input.GetKeyUp(KeyCode.Space))
            {
                StopAllCoroutines();
                StartCoroutine(Rotate(DOWN));
                StartCoroutine(Move(GetRandomPosition()));
            }
        }


        /// <summary>
        /// Rotate joint to the given local rotation
        /// </summary>
        IEnumerator Rotate(Quaternion rotation)
        {
            // balloonCollider.enabled = false;
            float counter = 0;
            float remaining = Quaternion.Angle(balloon.localRotation, rotation);
            while (remaining > Mathf.Epsilon)
            {
                balloon.localRotation = Quaternion.RotateTowards(balloon.localRotation, rotation, counter / .1f);
                remaining = Quaternion.Angle(balloon.localRotation, rotation);
                counter += Time.deltaTime;
                yield return null;
            }
            // balloonCollider.enabled = true;
            yield return null;
        }

        /// <summary>
        /// Move the container object to the given world location
        /// </summary>
        /// <param name="position"></param>
        /// <returns></returns>
        IEnumerator Move(Vector2 position)
        {
            balloonCollider.enabled = false;
            float counter = 0;
            Vector2 currentPos = transform.position;
            float remaining = (position - currentPos).sqrMagnitude;
            while (remaining > Mathf.Epsilon)
            {
                currentPos = Vector2.MoveTowards(currentPos, position, counter / 1f);
                remaining = (position - currentPos).sqrMagnitude;
                counter += Time.deltaTime;
                transform.position = currentPos;
                yield return null;
            }
            balloonCollider.enabled = true;
            yield return null;
        }

        /// <summary>
        /// Returns a random position on the balloon's random area
        /// </summary>
        /// <returns></returns>
        public Vector2 GetRandomPosition()
        {
            Vector2 pos = randomArea.localPosition;
            return pos + Random.insideUnitCircle * randomArea.lossyScale.z;
        }

        void OnDrawGizmos()
        {
            Gizmos.DrawWireSphere(randomArea.localPosition, randomArea.lossyScale.z);
        }
    }
}