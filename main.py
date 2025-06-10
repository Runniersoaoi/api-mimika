# main.py (FastAPI)
import cv2
import numpy as np
import pickle
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from io import BytesIO
from PIL import Image

app = FastAPI()

# CORS para conexión con Next.js
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173","*"],  # ajusta para seguridad
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'E', 2: 'I', 3: 'O', 4: 'U'}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(BytesIO(await file.read()))
        image = np.array(image.convert("RGB"))

        data_aux = []
        x_ = []
        y_ = []

        results = hands.process(image)

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

            if len(data_aux) == 42:
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict[int(prediction[0])]
                return {"prediction": predicted_character}

        return {"prediction": "None"}
    except Exception as e:
        print("Error:", e)
        return {"error": str(e)}
