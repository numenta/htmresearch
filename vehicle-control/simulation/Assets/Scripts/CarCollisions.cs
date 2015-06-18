using UnityEngine;
using System.Collections;

public class CarCollisions : MonoBehaviour {

	void OnCollisionEnter(Collision collision) {
		if (collision.gameObject.tag == "Boundary") {
			Application.LoadLevel(Application.loadedLevel);
		}
		else if (collision.gameObject.tag == "Finish") {
			Application.LoadLevel(Application.loadedLevel + 1);
		}
	}

}
