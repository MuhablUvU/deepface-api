from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from deepface import DeepFace
import numpy as np
import cv2

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Facial Emotion Recognition API is running"}

@app.post("/analyze/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        # Read the uploaded image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Analyze emotion using DeepFace
        result = DeepFace.analyze(img, actions = ['emotion'], enforce_detection=False)
        data = result[0]
        emotion_scores = {k: float(v) for k, v in data['emotion'].items()}

        return JSONResponse({
            "dominant_emotion": str(data['dominant_emotion']),
            "emotion_scores": emotion_scores
        })


    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
