/*
  ----------------------------------------------------------------------
  Numenta Platform for Intelligent Computing (NuPIC)
  Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
  with Numenta, Inc., for a separate license for this software code, the
  following terms and conditions apply:

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU Affero Public License version 3 as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU Affero Public License for more details.

  You should have received a copy of the GNU Affero Public License
  along with this program.  If not, see http://www.gnu.org/licenses.

  http://numenta.org/licenses/
  ----------------------------------------------------------------------
*/
using UnityEngine;
using System.Collections;

public class SweepSensor : MonoBehaviour {

	public LineRenderer lineRenderer;
	public int numRays = 120;
	public int graphVerticalOffset = 30;
	public float fieldOfView = 90;
	public float focalLength = 20;

	private bool lineIsActive = true;

	void Start () {
		lineRenderer.SetVertexCount(numRays + 1);
	}

	void Update () {
		RaycastHit hit;
		Quaternion rotation = Quaternion.Euler(0, fieldOfView / numRays, 0);
		Vector3 direction = transform.forward;

		float[] hits = new float[numRays];

		Vector3 translation = new Vector3(0, 0.02f, 0);
		Vector3 position;

		for (int i = 0; i < numRays; i++) {
			if (Physics.Raycast(transform.position, direction, out hit)) {
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
