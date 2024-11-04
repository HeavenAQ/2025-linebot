from flask import Flask, Request, Response, request, jsonify, send_file
import base64
import cv2
import os
import uuid
import numpy as np
from PoseModule import PoseDetector
from threading import Semaphore
from Types import COCOKeypoints

app = Flask(__name__)
pose_detector = PoseDetector()  # Initialize your PoseDetector class

USER = "admin"
PASSWORD = "thisisacomplicatedpassword"
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./output"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


class middleware:
    """
    simple wsgi middleware that check user name and password
    """

    def __init__(self, app):
        self.app = app
        self.username = USER
        self.password = PASSWORD

    def __call__(self, environ, start_response):
        request = Request(environ)
        if request.authorization is not None:
            # get username and password
            username = request.authorization["username"]
            password = request.authorization["password"]

            # check username and password
            if username == self.username and self.password == password:
                return self.app(environ, start_response)

        # if username and password is wrong
        return Response("Authorization Failed", 401, {"WWW-Authenticate": "Basic"})(
            environ,
            start_response,
        )


app = Flask(__name__)
app.wsgi_app = middleware(app.wsgi_app)

# use semaphore to limit the number of connections
max_connections = 10
sema = Semaphore(max_connections)


@app.before_request
def limit_connections() -> None | tuple[Response, int]:
    if not sema.acquire(blocking=False):
        return jsonify({"error": "Too many connections. Please try again later"}), 502


@app.after_request
def release_connection(response: Response):
    sema.release()
    return response


@app.route("/upload", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    # Save the video file
    file = request.files["video"]
    prefix = str(uuid.uuid4())
    filename = f"{prefix}_{file.filename}"
    file_path = f"{UPLOAD_FOLDER}/{filename}"
    file.save(file_path)

    # Process the video
    output_path, grading_score = process_video(file_path, filename)

    # Encode the processed video to base64
    with open(output_path, "rb") as f:
        video_data = f.read()
    video_base64 = base64.b64encode(video_data).decode("utf-8")

    # Return JSON with grading score and processed video
    return jsonify(
        {
            "grading_score": grading_score,
            "processed_video": video_base64,
        }
    ), 200


def process_video(video_path: str, out_filename: str) -> tuple[str, float]:
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_width, frame_height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    fps = cap.get(cv2.CAP_PROP_FPS)  # Get the FPS of the input video
    output_path = os.path.join(
        OUTPUT_FOLDER,
        out_filename,
    )

    # Define video writer to save processed video with the same FPS as input video
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    angle_sequences = []

    # Process each frame
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            break

        # Get pose estimation results
        results = pose_detector.get_pose(img)

        # Draw pose on the image
        pose_detector.show_pose(img, results)

        # Use pose detection to get landmarks and angle sequence for analysis
        landmarks = pose_detector.get_2d_landmarks(results)
        if landmarks:
            # Define the joints and their corresponding keypoints
            joints = {
                "Left Elbow": (
                    COCOKeypoints.LEFT_SHOULDER,
                    COCOKeypoints.LEFT_ELBOW,
                    COCOKeypoints.LEFT_WRIST,
                ),
                "Right Elbow": (
                    COCOKeypoints.RIGHT_SHOULDER,
                    COCOKeypoints.RIGHT_ELBOW,
                    COCOKeypoints.RIGHT_WRIST,
                ),
                "Left Knee": (
                    COCOKeypoints.LEFT_HIP,
                    COCOKeypoints.LEFT_KNEE,
                    COCOKeypoints.LEFT_ANKLE,
                ),
                "Right Knee": (
                    COCOKeypoints.RIGHT_HIP,
                    COCOKeypoints.RIGHT_KNEE,
                    COCOKeypoints.RIGHT_ANKLE,
                ),
                "Left Shoulder": (
                    COCOKeypoints.LEFT_HIP,
                    COCOKeypoints.LEFT_SHOULDER,
                    COCOKeypoints.LEFT_ELBOW,
                ),
                "Right Shoulder": (
                    COCOKeypoints.RIGHT_HIP,
                    COCOKeypoints.RIGHT_SHOULDER,
                    COCOKeypoints.RIGHT_ELBOW,
                ),
                "Left Hip": (
                    COCOKeypoints.LEFT_KNEE,
                    COCOKeypoints.LEFT_HIP,
                    COCOKeypoints.LEFT_SHOULDER,
                ),
                "Right Hip": (
                    COCOKeypoints.RIGHT_KNEE,
                    COCOKeypoints.RIGHT_HIP,
                    COCOKeypoints.RIGHT_SHOULDER,
                ),
                "Left Armpit": (
                    COCOKeypoints.LEFT_HIP,
                    COCOKeypoints.LEFT_SHOULDER,
                    COCOKeypoints.LEFT_ELBOW,
                ),
                "Right Armpit": (
                    COCOKeypoints.RIGHT_HIP,
                    COCOKeypoints.RIGHT_SHOULDER,
                    COCOKeypoints.RIGHT_ELBOW,
                ),
            }
            for joint_name, (point_a_id, point_b_id, point_c_id) in joints.items():
                # Check if all keypoints are detected
                if (
                    (point_a_id in landmarks)
                    and (point_b_id in landmarks)
                    and (point_c_id in landmarks)
                ):
                    point_a = landmarks[point_a_id]
                    point_b = landmarks[point_b_id]
                    point_c = landmarks[point_c_id]

                    # Compute the angle
                    angle = pose_detector.compute_angle(point_a, point_b, point_c)
                    if angle is not None:
                        # Show the angle on the image
                        pose_detector.show_angle_arc(
                            img, point_a, point_b, point_c, angle
                        )
                        pose_detector.logger.info(
                            f"{joint_name} Angle: {angle:.2f} degrees"
                        )
                    else:
                        pose_detector.logger.error(
                            f"Could not compute angle for {joint_name}"
                        )
                else:
                    pose_detector.logger.error(
                        f"Keypoints for {joint_name} not detected"
                    )
        else:
            pose_detector.logger.error("No landmarks detected")

        # Write the processed frame to output
        out.write(img)

    # Convert angle_sequences to numpy array for grading
    angle_sequences = np.array(angle_sequences)

    # Use pose detector's grading method (assuming you have it defined)
    # grading_score = pose_detector.grade_serve(angle_sequences)

    # Release resources
    cap.release()
    out.release()

    # Placeholder grading score
    grading_score = 85.0

    return output_path, grading_score


@app.route("/download/<filename>", methods=["GET"])
def download_file(filename):
    path = os.path.join(OUTPUT_FOLDER, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    else:
        return jsonify({"error": "File not found"}), 404


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=8000)
