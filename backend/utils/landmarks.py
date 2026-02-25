"""
Landmark Extraction and Normalization Module
Handles MediaPipe hand landmark detection and preprocessing
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List


class LandmarkExtractor:
    """Extracts and normalizes hand landmarks using MediaPipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize MediaPipe Hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,  # Only detect one hand
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def extract_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], bool]:
        """
        Extract hand landmarks from a frame
        
        Args:
            frame: BGR image from webcam
            
        Returns:
            Tuple of (normalized_landmarks, hand_detected)
            - normalized_landmarks: (63,) array or None
            - hand_detected: Boolean indicating if hand was found
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            # Get the first hand
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Extract raw landmarks (21 points × 3 coordinates = 63 values)
            raw_landmarks = []
            for landmark in hand_landmarks.landmark:
                raw_landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            raw_landmarks = np.array(raw_landmarks)
            
            # Normalize the landmarks
            normalized = self.normalize_landmarks(raw_landmarks)
            
            return normalized, True
        
        return None, False
    
    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize landmarks for scale and translation invariance
        
        Process:
        1. Reshape to (21, 3) for easier manipulation
        2. Make wrist-relative (subtract wrist coordinates)
        3. Scale by maximum distance (for depth invariance)
        4. Flatten back to (63,)
        
        Args:
            landmarks: Raw landmark array (63,)
            
        Returns:
            Normalized landmark array (63,)
        """
        # Reshape to (21, 3)
        points = landmarks.reshape(21, 3)
        
        # Step 1: Make wrist-relative (wrist is point 0)
        wrist = points[0].copy()
        points = points - wrist
        
        # Step 2: Calculate maximum distance for scaling
        distances = np.linalg.norm(points, axis=1)
        max_dist = np.max(distances)
        
        # Avoid division by zero
        if max_dist > 0:
            points = points / max_dist
        
        # Flatten back to (63,)
        return points.flatten()
    
    def draw_landmarks(self, frame: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Draw hand landmarks on the frame
        
        Args:
            frame: BGR image
            color: RGB color tuple (default: green)
            
        Returns:
            Frame with landmarks drawn
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        annotated_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return annotated_frame
    
    def close(self):
        """Release MediaPipe resources"""
        self.hands.close()


def landmarks_to_csv_row(landmarks: np.ndarray, label: str) -> str:
    """
    Convert landmarks and label to CSV row format
    
    Args:
        landmarks: Normalized landmarks (63,)
        label: Gesture label
        
    Returns:
        CSV row string
    """
    values = [label] + landmarks.tolist()
    return ','.join(map(str, values))


def get_csv_header() -> str:
    """
    Generate CSV header for landmark data
    
    Returns:
        CSV header string with label and 63 landmark features
    """
    headers = ['label']
    for i in range(21):
        headers.extend([f'lm{i}_x', f'lm{i}_y', f'lm{i}_z'])
    return ','.join(headers)
