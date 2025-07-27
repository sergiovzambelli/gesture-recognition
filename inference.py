import cv2
import torch
import numpy as np
import json
from collections import deque
from detector import HandVectorExtractor
from train_model import GestureRNNModel, LANDMARK_DIM, MAX_FRAMES, MAX_NUM_HANDS, DEVICE
from playsound import playsound
import sys

# Funzione per suono opzionale
def play_sound(audio):
    try:
        playsound(audio)
    except Exception as e:
        print(f"⚠️ Errore riproduzione audio: {e}")

# Mappa etichette
label_map = {
    "wave": 0,
    "no_wave": 1
}
inv_label_map = {v: k for k, v in label_map.items()}

# Caricamento modello
num_classes = len(label_map)
model = GestureRNNModel(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load("assets\\model.pth", map_location=DEVICE))
model.eval()

# Estrazione landmark e buffer
extractor = HandVectorExtractor(max_num_hands=MAX_NUM_HANDS)
buffer = deque(maxlen=MAX_FRAMES)
NO_HANDS_THRESHOLD = 10
no_hands_counter = 0

# Verifica validità del buffer
def buffer_is_valid(buffer, min_ratio=0.7):
    arr = np.array(buffer)
    if arr.shape[0] == 0:
        return False
    nonzero_frames = np.sum(~np.all(arr == 0, axis=1))
    return (nonzero_frames / arr.shape[0]) >= min_ratio

# Avvio webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Impossibile accedere alla webcam.")
    sys.exit(1)

font = cv2.FONT_HERSHEY_SIMPLEX
running = True

while running:
    ret, frame = cap.read()
    if not ret:
        print("❌ Errore nella lettura del frame dalla webcam. Esco.")
        break

    # Estrai vettore della mano
    vector = extractor.run(frame)

    if np.all(vector == 0):
        no_hands_counter += 1
    else:
        no_hands_counter = 0

    if no_hands_counter > NO_HANDS_THRESHOLD:
        print("[DEBUG] Mano non rilevata per molti frame, buffer svuotato.")
        buffer.clear()
        no_hands_counter = 0

    if vector.shape[0] != LANDMARK_DIM:
        padded = np.zeros(LANDMARK_DIM, dtype=np.float32)
        padded[: vector.shape[0]] = vector
        vector = padded

    buffer.append(vector)

    if len(buffer) == MAX_FRAMES and buffer_is_valid(buffer):
        sequence = np.array(buffer, dtype=np.float32)
        input_tensor = torch.from_numpy(sequence).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][pred_class].item()
            label = inv_label_map[pred_class]

        # Visualizzazione
        text = f"{label.upper()} ({confidence*100:.1f}%)"
        color = (0, 255, 0) if label == "wave" else (0, 0, 255)
        cv2.putText(frame, text, (10, 40), font, 1.0, color, 2)
        print(f"[PRED] {text}")
    else:
        print("[PRED] Buffer non valido o non pieno, nessuna predizione.")

    # Mostra il frame
    cv2.imshow("Wave Detection", frame)

    # Controlla le condizioni di uscita (tasto 'q' OPPURE chiusura finestra con 'X')
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or cv2.getWindowProperty("Wave Detection", cv2.WND_PROP_VISIBLE) < 1:
        if key == ord('q'):
            print("[INFO] Terminazione manuale con 'q'.")
        else:
            print("[INFO] Finestra chiusa con la X. Esco.")
        running = False # Imposta running a False per uscire dal ciclo
        # Il break è ridondante se si usa la flag 'running', ma lo lasciamo per chiarezza
        break

# Cleanup risorse
cap.release()
extractor.close()
cv2.destroyAllWindows()
print("[INFO] Risorse rilasciate. Programma terminato.")
sys.exit(0)