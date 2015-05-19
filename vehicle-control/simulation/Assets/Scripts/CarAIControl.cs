using UnityEngine;
using System.Collections;
using UnityStandardAssets.Vehicles.Car;

public class CarAIControl : MonoBehaviour {

	public CarUserControl userControl;

	void LateUpdate() {
		userControl.vertical = 1;

		if (API.instance.GetInput("steer") != null) {
			userControl.horizontal = (float)(double)API.instance.GetInput("steer");
		}
	}

}
