import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from collections import deque

from detector import HandVectorExtractor

# Import constants and model definition from your training script (model.py or main script)
from train_model import GestureRNNModel, LANDMARK_DIM, MAX_FRAMES, MAX_NUM_HANDS, DEVICE, NUM_LANDMARKS_PER_HAND

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from playsound import playsound


def play_sound(audio):
    try:
        playsound(audio)  # path to your MP3
    except Exception as e:
        print(f"⚠️ Failed to play MP3: {e}")


# === Hardcoded label map ===
label_map = {
    "wave": 1,
    "no_wave": 0
}
inv_label_map = {v: k for k, v in label_map.items()}
NUM_CLASSES = len(label_map)

# --- NORMALIZATION FUNCTIONS (COPIED FROM TRAINING SCRIPT) ---


def _normalize_single_hand(hand_landmarks: np.ndarray) -> np.ndarray:
    """Normalizza una singola mano per essere invariante a posizione, scala e rotazione."""
    if hand_landmarks.sum() == 0:  # Mano non rilevata
        return hand_landmarks

    # Reshape to (21, 2)
    landmarks = hand_landmarks.reshape(NUM_LANDMARKS_PER_HAND, 2)

    # 1. Traslazione: Centra sul polso (landmark 0)
    wrist_point = landmarks[0]
    translated_landmarks = landmarks - wrist_point

    # 2. Rotazione: Allinea la mano verticalmente
    # Usiamo il vettore dal polso (0) alla base del dito medio (9) as reference
    middle_finger_mcp = translated_landmarks[9]
    if np.all(middle_finger_mcp == 0):  # Rare case: wrist and mcp coincide
        return np.zeros_like(translated_landmarks).flatten()

    # Calculate angle to align the vector (0, -1) [negative y-axis]
    angle_to_rotate = np.arctan2(
        middle_finger_mcp[1], middle_finger_mcp[0]
    ) - np.deg2rad(-90)

    cos_a, sin_a = np.cos(angle_to_rotate), np.sin(angle_to_rotate)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    rotated_landmarks = np.dot(translated_landmarks, rotation_matrix.T)

    # 3. Scale: Normalize hand size
    # Use wrist-mcp distance as scale
    scale_dist = np.linalg.norm(rotated_landmarks[9])
    if scale_dist < 1e-6:
        return np.zeros_like(rotated_landmarks).flatten()

    normalized_landmarks = rotated_landmarks / scale_dist

    return normalized_landmarks.flatten()


def _normalize_landmarks_sequence(landmarks_sequence: np.ndarray) -> np.ndarray:
    """Applies advanced normalization to each hand in each frame."""
    if landmarks_sequence.size == 0:
        return landmarks_sequence

    normalized_seq = np.zeros_like(landmarks_sequence)

    num_frames = landmarks_sequence.shape[0]
    hand_dim = NUM_LANDMARKS_PER_HAND * 2

    for i in range(num_frames):
        frame_landmarks = landmarks_sequence[i]
        # Normalize the first hand
        normalized_seq[i, :hand_dim] = _normalize_single_hand(
            frame_landmarks[:hand_dim]
        )
        # Normalize the second hand (if present)
        if MAX_NUM_HANDS > 1:
            normalized_seq[i, hand_dim:] = _normalize_single_hand(
                frame_landmarks[hand_dim:]
            )

    return normalized_seq

# --- UTILITY TO CREATE METADATA FROM FOLDERS ---


def create_metadata_from_folders(root_dir: str, default_label: str = None) -> pd.DataFrame:
    data = []
    if not os.path.isdir(root_dir):
        print(f"Error: Directory not found: {root_dir}")
        return pd.DataFrame()

    # Check if root_dir itself contains video files directly
    video_files_in_root = [
        f for f in os.listdir(root_dir)
        if os.path.isfile(os.path.join(root_dir, f)) and f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
    ]

    if video_files_in_root:
        if default_label is None:
            # If no default label and videos are directly in root, they will be skipped
            pass
        else:
            for video_name in video_files_in_root:
                video_path = os.path.join(root_dir, video_name)
                data.append({"video_path": video_path, "label": default_label})
            return pd.DataFrame(data)

    # Original logic for class subfolders
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        for item_name in sorted(os.listdir(class_path)):
            item_path = os.path.join(class_path, item_name)
            if os.path.isfile(item_path) and item_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                data.append({"video_path": item_path, "label": class_name})

    return pd.DataFrame(data)


# --- MAIN INFERENCE LOGIC ---
if __name__ == "__main__":
    # Load model
    model = GestureRNNModel(num_classes=NUM_CLASSES)
    model_path = "assets\\model.pth"  # Or "final_gesture_rnn_assets\\model.pth"
    if not os.path.exists(model_path):
        print(
            f"Error: Model file not found at {model_path}. Please ensure training was successful and saved the model.")
        exit()

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    # Initialize extractor
    extractor = HandVectorExtractor(max_num_hands=MAX_NUM_HANDS)

    # Directory containing videos to be tested
    # This should point to the parent directory of your class folders (e.g., "clipped")
    # If you only have one folder like "clipped/wave" and want to assign all its videos
    # to "wave", then set INFERENCE_VIDEO_ROOT = "clipped/wave"
    # and adjust the create_metadata_from_folders call below.
    # Change this as per your actual directory structure
    INFERENCE_VIDEO_ROOT = "clipped"

    # Create metadata for inference videos
    # If INFERENCE_VIDEO_ROOT is like "clipped/wave" and all videos inside are "wave", use:
    # inference_df = create_metadata_from_folders(INFERENCE_VIDEO_ROOT, default_label="wave")
    inference_df = create_metadata_from_folders(
        INFERENCE_VIDEO_ROOT)  # Use this for multi-class structure

    if inference_df.empty:
        print(f"No videos found in '{INFERENCE_VIDEO_ROOT}'. Exiting.")
        exit()

    all_true_labels = []
    all_predicted_labels = []

    for idx, row in tqdm(inference_df.iterrows(), total=len(inference_df), desc="Processing videos"):
        video_path = row["video_path"]
        true_label_str = row["label"]
        true_label_idx = label_map[true_label_str]

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: Cannot open video file: {video_path}. Skipping.")
            continue

        sequence_landmarks = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks_vector = extractor.run(frame)
            if landmarks_vector.shape[0] != LANDMARK_DIM:
                padded_vec = np.zeros(LANDMARK_DIM, dtype=np.float32)
                padded_vec[: landmarks_vector.shape[0]] = landmarks_vector
                landmarks_vector = padded_vec
            sequence_landmarks.append(landmarks_vector)

        cap.release()

        if not sequence_landmarks:
            # This video yielded no valid landmarks, likely due to no hands detected
            # or very short length. It won't contribute to metrics unless explicitly handled.
            continue

        sequence_array = np.array(sequence_landmarks, dtype=np.float32)

        normalized_sequence = _normalize_landmarks_sequence(sequence_array)

        padded_input = np.zeros((MAX_FRAMES, LANDMARK_DIM), dtype=np.float32)
        seq_len = min(normalized_sequence.shape[0], MAX_FRAMES)
        if seq_len > 0:
            padded_input[:seq_len] = normalized_sequence[:seq_len]

        input_tensor = torch.from_numpy(padded_input).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.softmax(logits, dim=1)
            pred_class_idx = torch.argmax(probs, dim=1).item()

        all_true_labels.append(true_label_idx)
        all_predicted_labels.append(pred_class_idx)

    extractor.close()

    # --- Final Evaluation and Information Display ---
    if all_true_labels and all_predicted_labels:
        print("\n--- Inference Complete ---")
        play_sound("assets\\i_limoni.mp3")
        class_names = list(label_map.keys())

        # Overall Metrics
        overall_accuracy = accuracy_score(
            all_true_labels, all_predicted_labels)
        overall_precision = precision_score(
            all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
        overall_recall = recall_score(
            all_true_labels, all_predicted_labels, average='weighted', zero_division=0)
        overall_f1 = f1_score(
            all_true_labels, all_predicted_labels, average='weighted', zero_division=0)

        print(f"\nTotal videos processed: {len(all_true_labels)}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        print(f"Overall Precision (Weighted): {overall_precision:.4f}")
        print(f"Overall Recall (Weighted): {overall_recall:.4f}")
        print(f"Overall F1-Score (Weighted): {overall_f1:.4f}")

        # Classification Report (detailed per class)
        print("\n[ Detailed Classification Report ]")
        print(classification_report(all_true_labels, all_predicted_labels,
              target_names=class_names, digits=4, zero_division=0))

        # Confusion Matrix
        cm = confusion_matrix(all_true_labels, all_predicted_labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=[c.upper() for c in class_names],
            yticklabels=[c.upper() for c in class_names],
            cbar=False
        )
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

        # If it's a binary classification, you could also show ROC/AUC here.
        if NUM_CLASSES == 2:
            # Need to re-run predictions to get probabilities if not stored
            # For simplicity, if not needed often, you can regenerate this.
            # Otherwise, store all_probs from the loop.
            pass  # Skipping ROC for brevity, as probabilities are not stored in all_probs

    else:
        print("\nNo videos were successfully processed or no predictions could be made.")
        print("Please check your INFERENCE_VIDEO_ROOT path and video files.")
