from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from deepface import DeepFace
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# إعدادات الصورة
MAX_FILE_SIZE_MB = 5
RESIZE_IMAGE_TO = (224, 224)  # الحجم المثالي لمعظم موديلات DeepFace

@app.post("/analyze/")
async def analyze_emotion(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # فتح الصورة وتحويلها لـ RGB
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # لو الصورة حجمها أكبر من 5MB، قلل الجودة عشان نحافظ على الحجم
        if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
            # ضغط الصورة عن طريق حفظها بجودة أقل
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=70)
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")

        # Resize الصورة
        image = image.resize(RESIZE_IMAGE_TO)

        # تحويل الصورة لـ numpy ثم BGR
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # تحليل المشاعر باستخدام موديل Emotion (أدق من VGG-Face)
        result = DeepFace.analyze(img, actions=['emotion'], model_name='Emotion', enforce_detection=True)

        # تحويل القيم لفورمات JSON-friendly
        emotion_scores = {k: float(v) for k, v in result[0]['emotion'].items()}

        return JSONResponse({
            "dominant_emotion": str(result[0]['dominant_emotion']),
            "emotion_scores": emotion_scores
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
