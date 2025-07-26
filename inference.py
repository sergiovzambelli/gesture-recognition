import cv2
import torch
import numpy as np
import json
from collections import deque
from detector import HandVectorExtractor
from model import GestureRNNModel, LANDMARK_DIM, MAX_FRAMES, MAX_NUM_HANDS, DEVICE
from playsound import playsound

def play_sound(audio):
    try:
        playsound(audio)  # path to your MP3
    except Exception as e:
        print(f"⚠️ Failed to play MP3: {e}")

label_map = {
    "wave": 0,
    "no_wave": 1
}
inv_label_map = {v: k for k, v in label_map.items()}

# Invert again to get index-to-label mapping for display
inv_label_map = {idx: label for label, idx in label_map.items()}

# Load model
num_classes = len(label_map)
model = GestureRNNModel(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load("model.pth", map_location=DEVICE))
model.eval()

# Initialize extractor and rolling buffer
extractor = HandVectorExtractor(max_num_hands=MAX_NUM_HANDS)
buffer = deque(maxlen=MAX_FRAMES)

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Impossibile accedere alla webcam.")
    exit()

font = cv2.FONT_HERSHEY_SIMPLEX

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Errore nella lettura del frame dalla webcam.")
        break

    # Extract landmarks vector
    vector = extractor.run(frame)

    # Ensure vector has correct dimension, pad if necessary
    if vector.shape[0] != LANDMARK_DIM:
        padded = np.zeros(LANDMARK_DIM, dtype=np.float32)
        padded[: vector.shape[0]] = vector
        vector = padded

    buffer.append(vector)

    if len(buffer) == MAX_FRAMES:
        sequence = np.array(buffer, dtype=np.float32)
        input_tensor = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            label = inv_label_map[pred_class]

        # Display result on frame
        text = f"{label.upper()} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if label == "wave" else (0, 0, 255)
        cv2.putText(frame, text, (10, 40), font, 1.0, color, 2)

    cv2.imshow("Wave Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
extractor.close()
cv2.destroyAllWindows()
