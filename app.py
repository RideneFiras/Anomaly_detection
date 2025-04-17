from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import io
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import os

# Load your autoencoder model
model = load_model("autoencoder_model.h5", compile=False)
model.trainable = False

# FastAPI app
app = FastAPI()

# CORS (allows frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:5500"] if you want to restrict
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
@app.get("/")
def read_index():
    return FileResponse(os.path.join("static", "index.html"))
# Resize settings (must match your training settings!)
IMG_SIZE = (128, 128)

# Threshold for anomaly detection (you can tune this)
RECONSTRUCTION_THRESHOLD = 0.0148  # set based on validation error

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read and preprocess the image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize(IMG_SIZE)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Run the autoencoder
    reconstructed = model.predict(image_array)

    # Compute reconstruction error
    mse = np.mean(np.square(image_array - reconstructed))

    # Determine prediction
    prediction = "Defective" if mse > RECONSTRUCTION_THRESHOLD else "Good"

    # Confidence scaling
    max_mse_for_confidence = 0.05
    confidence = (1 - min(mse / max_mse_for_confidence, 1)) * 100

    # 🧪 Print for debugging
    print("====================================")
    print(f"MSE: {mse:.5f}")
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}%")
    print("====================================")

    return {
        "prediction": prediction,
        "confidence": round(confidence, 2)
    }
