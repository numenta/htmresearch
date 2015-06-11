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
