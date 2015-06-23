/*
  ----------------------------------------------------------------------
  Numenta Platform for Intelligent Computing (NuPIC)
  Copyright (C) 2015, Numenta, Inc.  Unless you have an agreement
  with Numenta, Inc., for a separate license for this software code, the
  following terms and conditions apply:

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License version 3 as
  published by the Free Software Foundation.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  See the GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see http://www.gnu.org/licenses.

  http://numenta.org/licenses/
  ----------------------------------------------------------------------
*/
using UnityEngine;
using System.Collections;

public class Reward : MonoBehaviour {

	public float reward = 100f;

	private bool collided = false;

	IEnumerator ResetLevel() {
		float lastSyncTime = API.instance.LastSyncTime();
		while (lastSyncTime == API.instance.LastSyncTime()) {
			yield return null;
		}

		Application.LoadLevel(Application.loadedLevel);
	}

	void OnCollisionEnter(Collision collision) {
		if (collided) return;

		API.instance.SetOutput("reward", reward);
		collided = true;

		StartCoroutine(ResetLevel());
	}

}
