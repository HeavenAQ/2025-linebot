import os
import uuid
import shutil
from threading import Semaphore
from typing import Annotated, Callable
from collections.abc import Awaitable

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, Request
from fastapi.responses import JSONResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from starlette.status import HTTP_401_UNAUTHORIZED

from Types import Handedness, Skill, VideoAnalysisResponse
from VideoProcessor import VideoProcessor


# FastAPI app initialization
app = FastAPI(title="Angle Analysis API", version="1.0.0")


# Configuration
# On Cloud Run, only /tmp is writable. Use it for transient data.
UPLOAD_FOLDER = "/tmp/uploads"
OUTPUT_FOLDER = "/tmp/output"
CSV_FILE = "/tmp/training_dataset.csv"  # CSV for storing training data (unused)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Semaphore to limit simultaneous connections
MAX_CONNECTIONS = 10
sema = Semaphore(MAX_CONNECTIONS)


# Basic Auth setup
security = HTTPBasic()
BASIC_AUTH_USERNAME = "admin"
BASIC_AUTH_PASSWORD = "thisisacomplicatedpassword"


def require_basic_auth(
    credentials: Annotated[HTTPBasicCredentials, Depends(security)],
) -> str:
    if not (
        credentials.username == BASIC_AUTH_USERNAME
        and credentials.password == BASIC_AUTH_PASSWORD
    ):
        raise HTTPException(
            status_code=HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username


@app.middleware("http")
async def limit_connections(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    # Prevent too many simultaneous connections
    if not sema.acquire(blocking=False):
        return JSONResponse(
            status_code=502,
            content={"error": "Too many connections. Please try again later"},
        )
    try:
        response = await call_next(request)
    finally:
        sema.release()
    return response


@app.post("/upload", response_model=VideoAnalysisResponse)
async def training_set(
    _: Annotated[str, Depends(require_basic_auth)],
    video: Annotated[UploadFile, File(...)],
    skill: Annotated[str, Form(...)],
    handedness: Annotated[str, Form(...)],
) -> VideoAnalysisResponse:
    """
    Endpoint to process a video and return grading and processed segment.
    """
    if not skill:
        raise HTTPException(status_code=400, detail="No skill provided")
    if not handedness:
        raise HTTPException(status_code=400, detail="No handedness provided")

    # Convert enums, with validation
    try:
        skill_enum = Skill.convert_to_enum(skill)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid skill value")
    try:
        handedness_enum = Handedness.convert_to_enum(handedness)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid handedness value")

    # Save the uploaded video
    prefix = str(uuid.uuid4())
    filename = f"{prefix}_{video.filename}"
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
    finally:
        await video.close()

    # Process the video using VideoProcessor
    processor = VideoProcessor(
        video_path=file_path, out_filename=filename, output_folder=OUTPUT_FOLDER
    )
    response = processor.process_video(skill_enum, handedness_enum)

    return response


@app.get("/health", response_model=dict[str, str])
async def health() -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "healthy"}


@app.get("/health/sec", response_model=dict[str, str])
async def health_sec(
    _: Annotated[str, Depends(require_basic_auth)],
) -> dict[str, str]:
    """
    Health check endpoint.
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0")
