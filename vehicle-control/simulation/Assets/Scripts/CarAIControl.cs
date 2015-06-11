using UnityEngine;
using System.Collections;
using UnityStandardAssets.Vehicles.Car;

public class CarAIControl : MonoBehaviour {

	public CarUserControl userControl;

	private float timeSinceReset;
	private float action;

	void ExecutePredefinedControl() {
		timeSinceReset += Time.deltaTime;

		if (timeSinceReset > 10 && timeSinceReset < 12) {
			userControl.horizontal = action;
		}
	}

	void Setup() {
		timeSinceReset = 0;
		action = Random.value > 0.5f ? -1 : 1;
	}

	void Start() {
		Setup();
	}

	void OnLevelWasLoaded(int level) {
		Setup();
	}

	void LateUpdate() {
		userControl.vertical = 1;

		if (API.instance.GetInput("steer") != null) {
			userControl.horizontal = (float)(double)API.instance.GetInput("steer");
		}

		userControl.horizontal = Mathf.Round(userControl.horizontal);

		ExecutePredefinedControl();

		API.instance.SetOutput("steer", userControl.horizontal);
	}

}
