import os
import cv2
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import List, Tuple

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

import subprocess
from sklearn.metrics import f1_score

from playsound import playsound


# Importa la classe corretta dal file vicino
# Assicurati che 'detector.py' sia nella stessa directory
from detector import HandVectorExtractor

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# === CONFIGURAZIONE E COSTANTI ===
DATA_ROOT = "frames"
MAX_FRAMES = 60
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "best_gesture_rnn_model.pth"

# --- CALCOLO DIMENSIONE VETTORE ---
# Questa configurazione assume che HandVectorExtractor restituisca tutti i 21 landmark
# per ogni mano, in un array piatto. La normalizzazione avanzata dipende da questo.
NUM_LANDMARKS_PER_HAND = 21
MAX_NUM_HANDS = 2
LANDMARK_DIM = MAX_NUM_HANDS * NUM_LANDMARKS_PER_HAND * 2  # (x,y) per ogni landmark


def play_sound(audio):
    try:
        playsound(audio)  # path to your MP3
    except Exception as e:
        print(f"⚠️ Failed to play MP3: {e}")


# === DATASET CON NORMALIZZAZIONE AVANZATA ===
class GestureLandmarkDataset(Dataset):
    """
    Dataset che carica sequenze di landmark, con normalizzazione avanzata
    e data augmentation. L'estrattore è inizializzato "lazy" per efficienza.
    """

    def __init__(self, dataframe: pd.DataFrame, label_map: dict, augment: bool = False):
        self.df = dataframe
        self.label_map = label_map
        self.augment = augment
        self.extractor = None

    def __len__(self) -> int:
        return len(self.df)

    def _normalize_single_hand(self, hand_landmarks: np.ndarray) -> np.ndarray:
        """Normalizza una singola mano per essere invariante a posizione, scala e rotazione."""
        if hand_landmarks.sum() == 0:  # Mano non rilevata
            return hand_landmarks

        # Reshape to (21, 2)
        landmarks = hand_landmarks.reshape(NUM_LANDMARKS_PER_HAND, 2)

        # 1. Traslazione: Centra sul polso (landmark 0)
        wrist_point = landmarks[0]
        translated_landmarks = landmarks - wrist_point

        # 2. Rotazione: Allinea la mano verticalmente
        # Usiamo il vettore dal polso (0) alla base del dito medio (9) come riferimento
        middle_finger_mcp = translated_landmarks[9]
        if np.all(middle_finger_mcp == 0):  # Caso raro: polso e mcp coincidono
            return np.zeros_like(translated_landmarks).flatten()

        # Calcola l'angolo per allineare il vettore (0, -1) [asse y negativo]
        angle_to_rotate = np.arctan2(
            middle_finger_mcp[1], middle_finger_mcp[0]
        ) - np.deg2rad(-90)

        cos_a, sin_a = np.cos(angle_to_rotate), np.sin(angle_to_rotate)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        rotated_landmarks = np.dot(translated_landmarks, rotation_matrix.T)

        # 3. Scala: Normalizza la dimensione della mano
        # Usiamo la distanza polso-mcp come scala
        scale_dist = np.linalg.norm(rotated_landmarks[9])
        if scale_dist < 1e-6:
            return np.zeros_like(rotated_landmarks).flatten()

        normalized_landmarks = rotated_landmarks / scale_dist

        return normalized_landmarks.flatten()

    def _normalize_landmarks(self, landmarks_sequence: np.ndarray) -> np.ndarray:
        """Applica la normalizzazione avanzata a ogni mano in ogni frame."""
        if landmarks_sequence.size == 0:
            return landmarks_sequence

        normalized_seq = np.zeros_like(landmarks_sequence)

        num_frames = landmarks_sequence.shape[0]
        hand_dim = NUM_LANDMARKS_PER_HAND * 2

        for i in range(num_frames):
            frame_landmarks = landmarks_sequence[i]
            # Normalizza la prima mano
            normalized_seq[i, :hand_dim] = self._normalize_single_hand(
                frame_landmarks[:hand_dim]
            )
            # Normalizza la seconda mano (se presente)
            if MAX_NUM_HANDS > 1:
                normalized_seq[i, hand_dim:] = self._normalize_single_hand(
                    frame_landmarks[hand_dim:]
                )

        return normalized_seq

    def _augment_landmarks(self, sequence: np.ndarray) -> np.ndarray:
        """Applica semplice data augmentation a una sequenza di landmark GIA' NORMALIZZATA."""
        if sequence.size == 0 or not self.augment:
            return sequence

        # Aggiungi un piccolo rumore Gaussiano
        noise = np.random.normal(0, 0.02, sequence.shape)
        augmented_sequence = sequence + noise

        # Applica un leggero time warping (semplice)
        if np.random.rand() > 0.5:
            stretch_factor = np.random.uniform(0.9, 1.1)
            orig_len = len(augmented_sequence)
            new_len = int(orig_len * stretch_factor)

            orig_indices = np.linspace(0, orig_len - 1, orig_len)
            new_indices = np.linspace(0, orig_len - 1, new_len)

            warped_sequence = np.zeros(
                (MAX_FRAMES, sequence.shape[1]), dtype=np.float32
            )
            for j in range(sequence.shape[1]):
                interpolated = np.interp(
                    new_indices, orig_indices, augmented_sequence[:, j]
                )
                if new_len >= MAX_FRAMES:
                    warped_sequence[:MAX_FRAMES, j] = interpolated[:MAX_FRAMES]
                else:
                    warped_sequence[:new_len, j] = interpolated

            return warped_sequence

        return augmented_sequence

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.extractor is None:
            # L'estrattore deve restituire tutti i 21 landmark
            self.extractor = HandVectorExtractor(max_num_hands=MAX_NUM_HANDS)

        video_info = self.df.iloc[idx]
        video_folder = video_info["frames_folder"]
        label_idx = self.label_map[video_info["label"]]

        sequence_landmarks = []
        try:
            frame_files = sorted(os.listdir(video_folder))
        except FileNotFoundError:
            print(
                f"ERRORE: La cartella {video_folder} non è stata trovata. Restituisce zero tensor."
            )
            return (
                torch.zeros((MAX_FRAMES, LANDMARK_DIM)),
                torch.LongTensor([0]).squeeze(),
            )

        frame_files_to_process = frame_files[:MAX_FRAMES]

        for frame_name in frame_files_to_process:
            frame_path = os.path.join(video_folder, frame_name)
            frame_image = cv2.imread(frame_path)
            if frame_image is None:
                continue

            landmarks_vector = self.extractor.run(frame_image)
            if landmarks_vector.shape[0] != LANDMARK_DIM:
                padded_vec = np.zeros(LANDMARK_DIM, dtype=np.float32)
                padded_vec[: landmarks_vector.shape[0]] = landmarks_vector
                landmarks_vector = padded_vec
            sequence_landmarks.append(landmarks_vector)

        if not sequence_landmarks:  # Se nessun frame è stato processato
            return (
                torch.zeros((MAX_FRAMES, LANDMARK_DIM)),
                torch.LongTensor([label_idx]).squeeze(),
            )

        sequence_array = np.array(sequence_landmarks, dtype=np.float32)

        normalized_sequence = self._normalize_landmarks(sequence_array)
        augmented_sequence = self._augment_landmarks(normalized_sequence)

        padded_sequence = np.zeros((MAX_FRAMES, LANDMARK_DIM), dtype=np.float32)
        seq_len = min(augmented_sequence.shape[0], MAX_FRAMES)
        if seq_len > 0:
            padded_sequence[:seq_len] = augmented_sequence[:seq_len]

        return (
            torch.from_numpy(padded_sequence),
            torch.LongTensor([label_idx]).squeeze(),
        )


# === MODELLO RNN ===
class GestureRNNModel(nn.Module):
    def __init__(
        self,
        input_dim=LANDMARK_DIM,
        rnn_hidden_dim=256,
        num_classes=2,
        rnn_layers=3,
        dropout_rate=0.5,
    ):
        super(GestureRNNModel, self).__init__()
        self.rnn = nn.LSTM(
            input_size=input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=rnn_layers,
            batch_first=True,
            dropout=dropout_rate if rnn_layers > 1 else 0,
            bidirectional=True,  # L'uso di un LSTM bidirezionale può catturare dipendenze temporali in entrambe le direzioni
        )
        # La dimensione dell'output della LSTM bidirezionale è 2 * rnn_hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(rnn_hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        rnn_out, _ = self.rnn(x)
        last_hidden = rnn_out[:, -1, :]
        return self.classifier(last_hidden)


# === FUNZIONI AUSILIARIE ===
def create_metadata_from_folders(root_dir: str) -> pd.DataFrame:
    print(f"Scansione della directory '{root_dir}'...")
    data = []
    if not os.path.isdir(root_dir):
        return pd.DataFrame()
    for class_name in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for video_name in sorted(os.listdir(class_path)):
            video_path = os.path.join(class_path, video_name)
            if os.path.isdir(video_path):
                data.append({"frames_folder": video_path, "label": class_name})
    return pd.DataFrame(data)


def train(model, train_loader, val_loader, optimizer, criterion, epochs, patience):
    print(f"\nInizio addestramento su {DEVICE}...")
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=3
    )

    for epoch in range(epochs):
        model.train()
        total_loss, all_preds, all_labels = 0, [], []

        for x, y in tqdm(
            train_loader, desc=f"Train Epoch {epoch+1}/{epochs}", unit="batch"
        ):
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

        train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, average="weighted")
        history["train_loss"].append(train_loss)
        history["train_f1"].append(train_f1)

        model.eval()
        total_loss, all_preds, all_labels = 0, [], []
        with torch.no_grad():
            for x, y in tqdm(
                val_loader, desc=f"Val Epoch {epoch+1}/{epochs} ", unit="batch"
            ):
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                total_loss += loss.item()
                all_preds.extend(out.argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss = total_loss / len(val_loader)
        val_f1 = f1_score(all_labels, all_preds, average="weighted")
        history["val_loss"].append(val_loss)
        history["val_f1"].append(val_f1)

        print(
            f"-> Epoch {epoch+1}: Train Loss={train_loss:.4f}, F1={train_f1:.4f} | Val Loss={val_loss:.4f}, F1={val_f1:.4f}"
        )
        play_sound("VAI_UOMO.mp3") 
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"   ✅ Modello salvato: Val Loss migliorata a {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(
                    f"   🛑 Early stopping! Val Loss non migliorata per {patience} epoche."
                )
                break

    return history


def evaluate_model(model, test_loader, label_map):
    """Valuta il modello sul test set e stampa le metriche di classificazione."""
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Testing Model", unit="batch"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)

            probs = torch.softmax(out, dim=1)
            preds = torch.argmax(out, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    class_names = list(label_map.keys())
    print("\n--- Risultati della Valutazione Finale su Test Set ---")

    # 1. Classification Report
    print("\n[ Classification Report ]")
    print(
        classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    )

    # 2. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Matrice di Confusione")
    plt.show()

    # 3. ROC Curve e AUC (specifico per classificazione binaria)
    if len(class_names) == 2:
        probs_class_1 = np.array(all_probs)[:, 1]
        fpr, tpr, _ = roc_curve(all_labels, probs_class_1)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(
            fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})"
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (ROC) Curve")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.show()


# === ESECUZIONE PRINCIPALE ===
if __name__ == "__main__":
    df = create_metadata_from_folders(DATA_ROOT)

    if df.empty:
        print(
            f"\nERRORE: Nessun dato trovato in '{DATA_ROOT}'. Controlla la struttura."
        )
    else:
        unique_labels = sorted(df["label"].unique())
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        num_classes = len(label_map)

        print(
            f"Trovate {len(df)} sequenze e {num_classes} classi: {list(label_map.keys())}"
        )
        print(f"Mappa delle etichette: {label_map}")

        print("\nAnalisi dello sbilanciamento del dataset...")
        class_counts = df["label"].value_counts()
        print(class_counts)

        # Calcola i pesi: weight = 1 / (frequenza della classe)
        # L'ordine deve corrispondere agli indici di label_map (0, 1, ...)
        class_weights = torch.tensor(
            [len(df) / class_counts[label] for label in unique_labels],
            dtype=torch.float32,
        ).to(DEVICE)

        print(f"Pesi calcolati per la loss function: {class_weights.cpu().numpy()}")
        # Esempio output: se class0 ha 40 campioni e class1 ha 60, i pesi saranno circa [2.5, 1.66]
        # Il modello sarà penalizzato 2.5 volte di più se sbaglia un campione di class0.

        # Suddivisione in Train (70%), Validation (15%), Test (15%)
        train_val_df, test_df = train_test_split(
            df, test_size=0.15, stratify=df["label"], random_state=42
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=(0.15 / 0.85),
            stratify=train_val_df["label"],
            random_state=42,
        )
        print(
            f"Dimensioni set: Training={len(train_df)}, Validation={len(val_df)}, Test={len(test_df)}"
        )

        # Creazione Datasets
        train_dataset = GestureLandmarkDataset(train_df, label_map, augment=True)
        val_dataset = GestureLandmarkDataset(val_df, label_map, augment=False)
        test_dataset = GestureLandmarkDataset(test_df, label_map, augment=False)

        # Creazione DataLoaders
        # Prova ad aumentare num_workers se hai più core CPU disponibili
        train_loader = DataLoader(
            train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )

        # Modello, Ottimizzatore, Criterio
        model = GestureRNNModel(
            num_classes=num_classes, rnn_hidden_dim=256, rnn_layers=3, dropout_rate=0.5
        ).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Addestramento
        history = train(
            model, train_loader, val_loader, optimizer, criterion, epochs=5, patience=10
        )
        print("\nAddestramento completato.")

        # Valutazione finale
        print(
            "\nCaricamento del modello con le migliori performance per la valutazione finale..."
        )
        model.load_state_dict(torch.load(MODEL_SAVE_PATH))

        evaluate_model(model, test_loader, label_map)

        # Salva i pesi finali del modello (dopo la valutazione)
        torch.save(model.state_dict(), "final_gesture_rnn_model.pth")
        print("\nPesi finali del modello salvati in 'final_gesture_rnn_model.pth'.")
        play_sound("i_limoni.mp3")
