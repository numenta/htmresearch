using UnityEngine;
using System.Collections;

public class WallAvoidance : MonoBehaviour {

	public CarAIControl carAIControl;
	public float fieldOfView = 180;
	public int numViews = 3;

	private GameObject closeWall;
	private GameObject closestWall;
	private float closestWallDistance = 10f;
	private float steeringDirection;
	private Vector3[] views;
	private float distance;

	private GameObject DetectWall() {
		RaycastHit hit;

		Quaternion rotation = Quaternion.Euler(0, fieldOfView / numViews, 0);
		Vector3 direction = Quaternion.Euler(0, -fieldOfView / 2f, 0) * transform.forward;

		closestWallDistance = 10f;
		closestWall = null;

		for (int i = 0; i < numViews; i++) {

			if (Physics.Raycast (transform.position, direction, out hit) &&
				hit.collider.gameObject.tag == "Boundary") {

				distance = Vector3.Distance(transform.position, hit.point);

				if (distance < closestWallDistance) {
					closestWall = hit.collider.gameObject;
					closestWallDistance = distance;
				}
			}
		
			direction = rotation * direction;
		}

		return closestWall;
	}

	private float AvoidanceTurningDirection(float collosionObjectAngle, float carAngle) {
		// Only change directions if moving towards the wall.
		float refAngle = Mathf.Abs (collosionObjectAngle - carAngle);

		if (refAngle < 90 || refAngle > 270) {
			if (collosionObjectAngle < carAngle || 360 - collosionObjectAngle < carAngle) {
				// The car is pointed towards the right, so turn it right.
				return 1f;
			}
			// The car is pointed towards the left, so turn it left.
			return -1f;
		}
		// The car is already moving away from the wall.
		return 0f;
	}

	private float HorizontalMoveAwayFromWall(GameObject wall) {

		var collisionY = wall.transform.rotation.eulerAngles.y;
		var carY = transform.rotation.eulerAngles.y;

		var steeringDirection = AvoidanceTurningDirection(collisionY, carY);

		return steeringDirection;
	}

	void Start() {
	}

	void Update() {
		
		closeWall = DetectWall();

		if (closeWall != null) {
			steeringDirection = HorizontalMoveAwayFromWall(closeWall);
			carAIControl.OverrideControl(steeringDirection, 1f);
		}

	}
	
}
