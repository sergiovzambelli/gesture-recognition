import cv2
import mediapipe as mp
import csv
import warnings
warnings.filterwarnings('ignore')

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

def absolute_value(hand_landmarks, frame_count):  
    tip_coords = {}
    tip_coords['frame'] = frame_count
    tip_idx = [4, 12, 20]
    wirst_c = hand_landmarks.landmark[0]
    for tip_id in tip_idx:
        lm = hand_landmarks.landmark[tip_id]
        tip_coords[f'{tip_id}_x'] = wirst_c.x - lm.x
        tip_coords[f'{tip_id}_y'] = wirst_c.y - lm.y
    print(tip_coords)
    return tip_coords

cap = cv2.VideoCapture('WAVE1.mp4')
tip_data = []
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    # Se hai landmark, disegnali
    if results.multi_hand_landmarks:
        print('iao')
        for _, hand_landmarks in enumerate(results.multi_hand_landmarks):
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if frame_count % 5 == 0:
                tip_data.append(absolute_value(hand_landmarks, frame_count))
    # Mostra il risultato (opzionale)
    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC per uscire
        break
print(tip_data)
cap.release()
cv2.destroyAllWindows()
