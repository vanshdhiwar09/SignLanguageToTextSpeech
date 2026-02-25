"""
Prediction Mediator Module
Handles prediction smoothing, buffering, and anti-flicker logic
"""

import numpy as np
from collections import deque
from typing import Optional, Dict, List
import time


class PredictionMediator:
    """
    Manages prediction stability through buffering and voting
    Prevents flickering and implements word construction logic
    """
    
    def __init__(
        self,
        buffer_size: int = 15,
        stability_threshold: int = 12,
        confidence_threshold: float = 0.85,
        hold_duration: float = 1.5
    ):
        """
        Initialize the mediator
        
        Args:
            buffer_size: Number of frames to keep in history
            stability_threshold: Minimum occurrences needed for stable prediction
            confidence_threshold: Minimum confidence to accept prediction
            hold_duration: Seconds to hold gesture before adding to sentence
        """
        self.buffer_size = buffer_size
        self.stability_threshold = stability_threshold
        self.confidence_threshold = confidence_threshold
        self.hold_duration = hold_duration
        
        # Prediction buffer (stores last N predictions)
        self.prediction_buffer = deque(maxlen=buffer_size)
        
        # Current stable prediction
        self.current_prediction = "NOTHING"
        self.current_confidence = 0.0
        
        # Word construction
        self.sentence = ""
        self.last_stable_gesture = "NOTHING"
        self.gesture_start_time = None
        self.gesture_added = False
    
    def add_prediction(self, prediction: str, confidence: float) -> Dict:
        """
        Add a new prediction and compute stable output
        
        Args:
            prediction: Predicted gesture label
            confidence: Prediction confidence (0-1)
            
        Returns:
            Dictionary with current state:
            - stable_prediction: The stable gesture
            - confidence: Confidence value
            - sentence: Current sentence
            - buffer_status: Buffer fill percentage
        """
        # Only add if confidence is above threshold
        if confidence >= self.confidence_threshold:
            self.prediction_buffer.append(prediction)
        else:
            self.prediction_buffer.append("NOTHING")
        
        # Compute stable prediction through voting
        stable_pred = self._compute_stable_prediction()
        
        # Update current state
        self.current_prediction = stable_pred
        self.current_confidence = confidence
        
        # Handle word construction
        self._update_sentence(stable_pred)
        
        return {
            "stable_prediction": self.current_prediction,
            "confidence": self.current_confidence,
            "sentence": self.sentence,
            "buffer_status": len(self.prediction_buffer) / self.buffer_size,
            "raw_prediction": prediction
        }
    
    def _compute_stable_prediction(self) -> str:
        """
        Compute stable prediction using majority voting
        
        Returns:
            Most common prediction if it meets stability threshold
        """
        if len(self.prediction_buffer) < self.buffer_size:
            return "NOTHING"
        
        # Count occurrences
        predictions = list(self.prediction_buffer)
        unique, counts = np.unique(predictions, return_counts=True)
        
        # Find most common prediction
        max_idx = np.argmax(counts)
        most_common = unique[max_idx]
        count = counts[max_idx]
        
        # Check if it meets stability threshold
        if count >= self.stability_threshold:
            return most_common
        
        return "NOTHING"
    
    def _update_sentence(self, stable_prediction: str):
        """
        Update sentence based on stable prediction and hold duration
        
        Args:
            stable_prediction: Current stable gesture
        """
        current_time = time.time()
        
        # Check if gesture changed
        if stable_prediction != self.last_stable_gesture:
            self.last_stable_gesture = stable_prediction
            self.gesture_start_time = current_time
            self.gesture_added = False
            return
        
        # If gesture is stable and not yet added
        if stable_prediction != "NOTHING" and not self.gesture_added:
            if self.gesture_start_time is None:
                self.gesture_start_time = current_time
                return
            
            # Check if held long enough
            hold_time = current_time - self.gesture_start_time
            if hold_time >= self.hold_duration:
                self._add_to_sentence(stable_prediction)
                self.gesture_added = True
    
    def _add_to_sentence(self, gesture: str):
        """
        Add gesture to sentence with special handling
        
        Args:
            gesture: Gesture to add
        """
        # Handle special gestures
        if gesture == "SPACE":
            self.sentence += " "
        elif gesture == "DELETE":
            self.sentence = self.sentence[:-1]
        elif gesture == "NOTHING":
            pass  # Do nothing
        else:
            # Check if it's a word or letter
            if gesture in ["HELLO", "YES", "NO", "THANKYOU", "ILOVEYOU", "STOP"]:
                # Add word with space
                if self.sentence and not self.sentence.endswith(" "):
                    self.sentence += " "
                self.sentence += gesture
                self.sentence += " "
            else:
                # Add single letter
                self.sentence += gesture
    
    def clear_sentence(self):
        """Clear the current sentence"""
        self.sentence = ""
        self.gesture_added = False
    
    def get_sentence(self) -> str:
        """Get the current sentence"""
        return self.sentence
    
    def reset(self):
        """Reset all state"""
        self.prediction_buffer.clear()
        self.current_prediction = "NOTHING"
        self.current_confidence = 0.0
        self.sentence = ""
        self.last_stable_gesture = "NOTHING"
        self.gesture_start_time = None
        self.gesture_added = False
    
    def clear_buffer(self):
        """Clear prediction buffer for immediate reset"""
        self.prediction_buffer.clear()
        self.current_prediction = "NOTHING"
        self.last_stable_gesture = "NOTHING"
        self.gesture_start_time = None
        self.gesture_added = False
    
    def get_buffer_distribution(self) -> Dict[str, int]:
        """
        Get distribution of predictions in buffer
        
        Returns:
            Dictionary mapping gesture to count
        """
        if not self.prediction_buffer:
            return {}
        
        predictions = list(self.prediction_buffer)
        unique, counts = np.unique(predictions, return_counts=True)
        
        return dict(zip(unique, counts))
