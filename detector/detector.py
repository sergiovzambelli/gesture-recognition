import cv2
import mediapipe as mp
import numpy as np
from typing import List, Any, Optional

class HandVectorExtractor:
    """
    Estrae un singolo vettore di coordinate 2D relative al polso per i polpastrelli specificati.
    """
    def __init__(self,
                 static_image_mode: bool = False,
                 max_num_hands: int = 2,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 tip_indices: Optional[List[int]] = None):
        """
        Inizializza il rilevatore di mani di MediaPipe.
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.tip_indices = tip_indices if tip_indices is not None else [4, 12, 20]
        
        # Calculate the size of the block for a single hand's landmarks
        self.single_hand_vector_size = len(self.tip_indices) * 2
        
        # Calculate the total expected output vector size
        self.max_num_hands = max_num_hands
        self.output_vector_size = self.max_num_hands * self.single_hand_vector_size

        # Defensive check: if tip_indices is empty, this could cause issues.
        # Although your current setup prevents this.
        if self.single_hand_vector_size == 0:
            print("WARNING: tip_indices is empty. Hand vector size will be 0, leading to empty vectors.")


    def run(self, image: Any) -> np.ndarray:
        """
        Estrae i dati e li restituisce come un singolo vettore NumPy.
        Returns a zero-filled vector of self.output_vector_size if no hands are detected
        or if an issue occurs during landmark extraction for a hand.
        """
        # Always initialize the output_vector to the correct final size, filled with zeros.
        # This is the key to preventing the "broadcast shape (0,)" error when no hands are found
        # or when a hand's data cannot be properly extracted.
        output_vector = np.zeros(self.output_vector_size, dtype=np.float32)

        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False # For MediaPipe performance
        
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Ensure we don't exceed the configured max_num_hands
                if i >= self.max_num_hands:
                    break

                wrist = hand_landmarks.landmark[0]
                hand_coords = []
                
                # Extract coordinates for specified tip indices, relative to wrist
                for tip_id in self.tip_indices:
                    # Basic bounds check for tip_id, though MediaPipe usually ensures valid indices
                    if 0 <= tip_id < len(hand_landmarks.landmark):
                        lm = hand_landmarks.landmark[tip_id]
                        hand_coords.extend([lm.x - wrist.x, lm.y - wrist.y])
                    else:
                        # This should ideally not happen if tip_indices are standard MediaPipe indices
                        print(f"WARNING: Invalid tip_id {tip_id} encountered for a hand. Skipping this tip.")
                        # Append zeros for this invalid tip to maintain vector size
                        hand_coords.extend([0.0, 0.0])

                # Convert hand_coords list to a numpy array for shape checking and assignment
                hand_coords_array = np.array(hand_coords, dtype=np.float32)

                # Calculate start and end indices for this hand's data
                start_idx = i * self.single_hand_vector_size
                end_idx = start_idx + self.single_hand_vector_size

                # --- CRITICAL FIX PART ---
                # Ensure the extracted hand_coords_array has the expected size before assigning.
                # If it doesn't, it implies an issue (e.g., missing tip_id or internal problem).
                # In such cases, we'll leave the corresponding block in output_vector as zeros.
                if hand_coords_array.shape[0] == self.single_hand_vector_size:
                    output_vector[start_idx:end_idx] = hand_coords_array
                else:
                    # Log a warning if the extracted data size doesn't match expectations
                    print(f"WARNING: Hand {i} extracted data size mismatch. Expected {self.single_hand_vector_size}, Got {hand_coords_array.shape[0]}. Filling with zeros for this hand.")
                    # output_vector for this hand's block will remain zeros as it was initialized that way.

        return output_vector

    def close(self):
        self.hands.close()