using UnityEngine;
using System.Collections;

public class SweepSensor : MonoBehaviour {

	public LineRenderer lineRenderer;
	public int numRays = 120;
	public int graphVerticalOffset = 30;
	public float fieldOfView = 90;
	public float focalLength = 20;

	private bool lineIsActive = true;
	private bool holdingKey = false;
	private string screenshotName = "screenshot.png";

	void Start () {
		lineRenderer.SetVertexCount(numRays + 1);
	}

	void Update () {
		RaycastHit hit;
		Quaternion rotation = Quaternion.Euler(0, fieldOfView / numRays, 0);
		Vector3 direction = Quaternion.Euler(0, -fieldOfView / 2f, 0) * transform.forward;

		float[] hits = new float[numRays];

		Vector3 translation = new Vector3(0, 0.02f, 0);
		Vector3 position;

		for (int i = 0; i < numRays; i++) {
			if (Physics.Raycast(transform.position, direction, out hit) &&
			    hit.collider.gameObject.tag != "Boundary") {
				hits[i] = Mathf.Lerp(1, 0, hit.distance / focalLength);

				position = transform.position + direction;
				position += translation * (graphVerticalOffset - hit.distance);
			}
			else {
				hits[i] = 0;

				position = transform.position;
			}

			lineRenderer.SetPosition(i, position);
			if (i == 0) {
				lineRenderer.SetPosition(numRays, position);
			}

			direction = rotation * direction;
		}

		if (Input.GetKey (KeyCode.C)) {
			// For collecting depth data.
			API.instance.SetOutput("collectKeyPressed", 1);
			holdingKey = true;
		}
		else {
			if (holdingKey) {
				screenshotName = "screenshot_"+System.DateTime.Now.ToString("%y%m%d%H%M%s")+".png";
				Application.CaptureScreenshot("Screenshots/"+screenshotName);
				holdingKey = false;
			}

			API.instance.SetOutput("collectKeyPressed", 0);
		}

		API.instance.SetOutput(gameObject.name, hits);

		if (Input.GetKeyDown("space")) {
			lineIsActive = !lineIsActive;
		}

		lineRenderer.SetWidth(0, 0);
		if (lineIsActive) {
			lineRenderer.SetWidth(0.01f, 0.01f);  // TODO: refactor
		}
	}

}
