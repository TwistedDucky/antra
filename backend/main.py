from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import os
import uuid
import shutil

from .tracker import analyze_video


app = FastAPI(
    title="ANTRA",
    description="Strava for Ants 🐜",
    version="0.1.0"
)


# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


UPLOAD_DIR = "uploads"

os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


@app.get("/")
def home():

    return {
        "name": "ANTRA",
        "message": "Strava for Ants 🐜",
        "status": "running"
    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...)
):

    # Check file type
    allowed_types = [
        "video/mp4",
        "video/quicktime",
        "video/x-msvideo",
        "video/webm"
    ]

    if video.content_type not in allowed_types:

        raise HTTPException(
            status_code=400,
            detail="Please upload a video file."
        )

    # Generate unique filename
    extension = os.path.splitext(
        video.filename
    )[1]

    filename = (
        f"{uuid.uuid4()}{extension}"
    )

    video_path = os.path.join(
        UPLOAD_DIR,
        filename
    )

    # Save uploaded video
    try:

        with open(
            video_path,
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                video.file,
                buffer
            )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Could not save video: {str(e)}"
        )

    # Run computer vision
    try:

        statistics = analyze_video(
            video_path
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {str(e)}"
        )

    finally:

        # Delete uploaded video
        if os.path.exists(video_path):

            os.remove(video_path)

    return statistics