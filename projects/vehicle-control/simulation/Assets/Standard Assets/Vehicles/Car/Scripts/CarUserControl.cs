using System;
using UnityEngine;
using UnityStandardAssets.CrossPlatformInput;

namespace UnityStandardAssets.Vehicles.Car
{
    [RequireComponent(typeof (CarController))]
    public class CarUserControl : MonoBehaviour
    {
        public float horizontal = 0;
        public float vertical = 0;

        private CarController m_Car; // the car controller we want to use


        private void Awake()
        {
            // get the car controller
            m_Car = GetComponent<CarController>();
        }

        private void Update()
        {
            horizontal = CrossPlatformInputManager.GetAxis("Horizontal");
            vertical = CrossPlatformInputManager.GetAxis("Vertical");
        }

        private void FixedUpdate()
        {
            // pass the input to the car!
#if !MOBILE_INPUT
            float handbrake = CrossPlatformInputManager.GetAxis("Jump");
            m_Car.Move(horizontal, vertical, vertical, handbrake);
#else
            m_Car.Move(horizontal, vertical, vertical, 0f);
#endif
        }
    }
}
