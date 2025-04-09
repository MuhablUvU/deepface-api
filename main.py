import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
from starlette.middleware.cors import CORSMiddleware

app = FastAPI(title="DeepFace Emotion Detection API")

# CORS middleware (adjust settings as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Verify file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    try:
        # Read file and decode image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Couldn't decode the image.")

        # Analyze image using DeepFace with emotion action
        result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

        # If result is returned as a list (multiple faces), pick the first one.
        if isinstance(result, list):
            result = result[0]

        return JSONResponse(content={
            "predicted_emotion": result["dominant_emotion"],
            "all_emotions": result["emotion"]
        })

    except Exception as e:
        print("[ERROR]", e)
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
