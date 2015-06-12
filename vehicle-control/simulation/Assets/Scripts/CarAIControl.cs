using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityStandardAssets.Vehicles.Car;

public class CarAIControl : MonoBehaviour {

	public CarUserControl userControl;

	private float timeSinceReset;
	private float action;

	void UpdateControl() {
		userControl.vertical = 1;

		if (API.instance.GetInput("steer") != null) {
			userControl.horizontal = (float)(double)API.instance.GetInput("steer");
		}

		userControl.horizontal = Mathf.Round(userControl.horizontal);

		ExecutePredefinedControl();

		API.instance.SetOutput("steer", userControl.horizontal);
	}

	void UpdateQValues() {
		Dictionary<string, object> qValues = (Dictionary<string, object>)API.instance.GetInput("qValues");
		if (qValues == null) return;

		double qLeft = (double)qValues["-1"];
		double qStraight = (double)qValues["0"];
		double qRight = (double)qValues["1"];

		int lineLength = 10;

		Debug.DrawLine(transform.position, transform.position + 10 * -transform.right, ColorForQValue(qLeft), 0, false);
		Debug.DrawLine(transform.position, transform.position + 10 * transform.right, ColorForQValue(qRight), 0, false);
		Debug.DrawLine(transform.position, transform.position + 10 * transform.forward, ColorForQValue(qStraight), 0, false);
	}

	Color ColorForQValue(double qValue) {
		float x = -0.5f * (float)qValue;
		float t = 1 / (1 + Mathf.Exp(x));
		return Color.Lerp(Color.red, Color.green, t);
	}

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

	IEnumerator Start() {
		Setup();

		while (true) {
			UpdateControl();
			UpdateQValues();
			yield return null;
		}
	}

	void OnLevelWasLoaded(int level) {
		Setup();
	}

}
