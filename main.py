from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from deepface import DeepFace
import os
from io import BytesIO
from PIL import Image

app = FastAPI()

class EmotionResponse(BaseModel):
    emotion: str

@app.post("/predict_emotion", response_model=EmotionResponse)
async def predict_emotion(file: UploadFile = File(...)):
    try:
        # Read the image file
        image_bytes = await file.read()
        image = Image.open(BytesIO(image_bytes))
        
        # Save image to a temporary file
        image_path = "temp.jpg"
        image.save(image_path)

        # Use DeepFace to analyze the image
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'])
        emotion = result[0]['dominant_emotion']

        # Clean up the temporary file
        os.remove(image_path)
        
        return EmotionResponse(emotion=emotion)
    
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
