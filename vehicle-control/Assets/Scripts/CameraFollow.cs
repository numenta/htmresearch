using UnityEngine;
using System.Collections;

public class CameraFollow : MonoBehaviour {

	public Vector3 cameraFollowOffset = new Vector3(0, 3, -10);

	void Update () {
		Camera.main.transform.position = transform.position + cameraFollowOffset;
	}

}
