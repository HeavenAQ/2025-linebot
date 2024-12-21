from flask import Flask, request, jsonify
import os
import uuid
from threading import Semaphore
from Middleware import Middleware
from Types import Handedness, Skill
from VideoProcessor import VideoProcessor  # Your VideoProcessor class

# Flask app initialization
app = Flask(__name__)
app.wsgi_app = Middleware(app.wsgi_app)


# Configuration
UPLOAD_FOLDER = "./uploads"
OUTPUT_FOLDER = "./output"
CSV_FILE = "./training_dataset.csv"  # CSV for storing training data
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Semaphore to limit simultaneous connections
MAX_CONNECTIONS = 10
sema = Semaphore(MAX_CONNECTIONS)


@app.before_request
def limit_connections():
    """
    Prevent too many simultaneous connections.
    """
    if not sema.acquire(blocking=False):
        return jsonify({"error": "Too many connections. Please try again later"}), 502


@app.after_request
def release_connection(response):
    """
    Release the semaphore after the request is processed.
    """
    sema.release()
    return response


@app.route("/upload", methods=["POST"])
def training_set():
    """
    Endpoint to process a video and save training angles data.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    # Ensure the parameters are given correctly
    if not (file := request.files.get("video")):
        return jsonify({"error": "No video file provided"}), 400
    if not (skill := request.form.get("skill")):
        return jsonify({"error": "No skill provided"}), 400
    if not (handedness := request.form.get("handedness")):
        return jsonify({"error": "No handedness provided"}), 400

    # Preprocess the paramteres
    skill = Skill.convert_to_enum(skill)
    handedness = Handedness.convert_to_enum(handedness)

    # Save the uploaded video
    prefix = str(uuid.uuid4())
    filename = f"{prefix}_{file.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    # Process the video using VideoProcessor
    processor = VideoProcessor(
        video_path=file_path, out_filename=filename, output_folder=OUTPUT_FOLDER
    )
    response = processor.process_video(skill, handedness)
    print(response["grade"])

    # Response
    return jsonify(response), 200


@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """
    return jsonify({"status": "healthy"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=8000)
