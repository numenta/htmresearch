using UnityEngine;
using System.Collections;

public class Reward : MonoBehaviour {

	public float reward = 100f;

	private bool collided = false;

	IEnumerator ResetLevel() {
		yield return new WaitForSeconds(1.0f);
		Application.LoadLevel(Application.loadedLevel);
	}

	void OnCollisionEnter(Collision collision) {
		if (collided) return;

		API.instance.SetOutput("reward", reward);
		collided = true;

		StartCoroutine(ResetLevel());
	}

}
