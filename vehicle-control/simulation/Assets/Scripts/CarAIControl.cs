﻿using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityStandardAssets.Vehicles.Car;

public class CarAIControl : MonoBehaviour {

	public CarUserControl userControl;

	private float timeSinceReset;
	private float action;
	private float overrideHorizontal;
	private float overrideVertical;
	private bool directionIsOverridden = false;

	public void OverrideControl(float horizontal, float vertical) {
		overrideHorizontal = horizontal;
		overrideVertical = vertical;
		directionIsOverridden = true;
	}

	void UpdateControl() {
		userControl.vertical = 1;

		if (API.instance.GetInput("steer") != null) {
			userControl.horizontal = (float)(double)API.instance.GetInput("steer");
		}

		userControl.horizontal = Mathf.Round(userControl.horizontal);

		if (directionIsOverridden) {
			userControl.horizontal = overrideHorizontal;
			userControl.vertical = overrideVertical;
			directionIsOverridden = false;
		}

		ExecutePredefinedControl();

		API.instance.SetOutput("steer", userControl.horizontal);
	}

	void UpdateQValues() {
		Dictionary<string, object> qValues = (Dictionary<string, object>)API.instance.GetInput("qValues");
		if (qValues == null) return;

		double qLeft = (double)qValues["-1"];
		double qStraight = (double)qValues["0"];
		double qRight = (double)qValues["1"];

		DrawLineAtCar(-transform.right, ColorForQValue(qLeft));
		DrawLineAtCar(transform.forward, ColorForQValue(qStraight));
		DrawLineAtCar(transform.right, ColorForQValue(qRight));

		string bestAction = (string)API.instance.GetInput("bestAction");
		if (bestAction == null) return;

		if (bestAction == "-1") DrawLineBehindCar(-transform.right, ColorForQValue(qLeft));
		if (bestAction == "0") DrawLineBehindCar(transform.forward, ColorForQValue(qStraight));
		if (bestAction == "1") DrawLineBehindCar(transform.right, ColorForQValue(qRight));
	}

	Color ColorForQValue(double qValue) {
		float x = -0.5f * (float)qValue;
		float t = 1 / (1 + Mathf.Exp(x));
		return Color.Lerp(Color.red, Color.green, t);
	}

	void DrawLineAtCar(Vector3 direction, Color color) {
		int lineLength = 10;
		Debug.DrawLine(transform.position, transform.position + lineLength * direction, color, 0, false);
	}

	void DrawLineBehindCar(Vector3 direction, Color color) {
		int lineLength = 4;
		int distance = 5;

		Vector3 position = transform.position - distance * transform.forward;
		Debug.DrawLine(position, position + lineLength * direction, color, 0, false);
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
