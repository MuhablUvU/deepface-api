import os
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from typing import List, Optional
from pydantic import BaseModel
from deepface import DeepFace
import shutil
from pathlib import Path

# Create the FastAPI app
app = FastAPI(title="Face Recognition API", 
              description="Face recognition service using DeepFace",
              version="1.0.0")

# Configuration
UPLOAD_FOLDER = Path("uploads")
DATABASE_FOLDER = Path("database")
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create directories if they don't exist
UPLOAD_FOLDER.mkdir(exist_ok=True)
DATABASE_FOLDER.mkdir(exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Response models
class RecognitionResult(BaseModel):
    identity: str
    confidence: float

class RecognitionResponse(BaseModel):
    results: List[RecognitionResult]
    message: str

# Helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html", "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

@app.post("/recognize", response_model=RecognitionResponse)
async def recognize_face(file: UploadFile = File(...), threshold: float = Form(0.6)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File format not supported")
    
    # Save the uploaded file
    file_path = UPLOAD_FOLDER / secure_filename(file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Get all database images
    db_images = [str(DATABASE_FOLDER / f) for f in os.listdir(DATABASE_FOLDER) 
                if allowed_file(f)]
    
    if not db_images:
        return JSONResponse(
            status_code=200,
            content={"results": [], "message": "No faces in database to compare with"}
        )
    
    try:
        # Perform recognition
        results = DeepFace.find(
            img_path=str(file_path),
            db_path=str(DATABASE_FOLDER),
            model_name="VGG-Face",
            distance_metric="cosine",
            enforce_detection=False
        )
        
        recognition_results = []
        for _, row in results[0].iterrows():
            if row["VGG-Face_cosine"] <= threshold:
                identity = Path(row["identity"]).stem
                confidence = 1 - row["VGG-Face_cosine"]  # Convert distance to confidence
                recognition_results.append(
                    RecognitionResult(identity=identity, confidence=float(confidence))
                )
        
        message = "Faces found" if recognition_results else "No matching faces found"
        return RecognitionResponse(results=recognition_results, message=message)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during recognition: {str(e)}")
    finally:
        # Clean up
        if file_path.exists():
            os.remove(file_path)

@app.post("/register")
async def register_face(file: UploadFile = File(...), name: str = Form(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="File format not supported")
    
    # Create a filename with the person's name
    filename = f"{secure_filename(name)}_{secure_filename(file.filename)}"
    file_path = DATABASE_FOLDER / filename
    
    # Save the file to the database folder
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Verify that the image contains a detectable face
    try:
        DeepFace.extract_faces(str(file_path), enforce_detection=True)
        return JSONResponse(
            status_code=200,
            content={"message": f"Face registered successfully as {name}"}
        )
    except Exception as e:
        # If no face detected, remove the file and return error
        if file_path.exists():
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"No face detected in image: {str(e)}")

def secure_filename(filename):
    """Make a filename secure, removing suspicious characters"""
    return "".join(c for c in filename if c.isalnum() or c in "._- ").strip()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", 8000)), reload=True)