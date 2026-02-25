"""
Data Collection Tool for Sign Language Gestures
Records hand landmarks for training the model
"""

import cv2
import os
import sys
from pathlib import Path

# Add backend directory to Python path for imports
backend_dir = Path(__file__).parent.absolute()
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from utils.landmarks import LandmarkExtractor, landmarks_to_csv_row, get_csv_header


class DataCollector:
    """Interactive tool for collecting gesture data"""
    
    def __init__(self, output_file: str = "data/gesture_data.csv"):
        self.output_file = output_file
        self.extractor = LandmarkExtractor()
        self.cap = None
        
        # Create data directory if it doesn't exist
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Initialize CSV file with header if it doesn't exist
        if not os.path.exists(output_file):
            with open(output_file, 'w') as f:
                f.write(get_csv_header() + '\n')
            print(f"✅ Created new dataset file: {output_file}")
        else:
            print(f"📂 Using existing dataset file: {output_file}")
    
    def start_camera(self):
        """Initialize webcam"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Could not open webcam")
            sys.exit(1)
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized")
    
    def collect_gesture(self, label: str, num_samples: int = 500):
        """
        Collect samples for a specific gesture
        
        Args:
            label: Gesture label (e.g., 'A', 'HELLO')
            num_samples: Number of samples to collect
        """
        print(f"\n🎯 Collecting data for gesture: {label}")
        print(f"📊 Target samples: {num_samples}")
        print("\n⚠️  Instructions:")
        print("   - Position your hand in the camera view")
        print("   - Press SPACE to start recording")
        print("   - Hold the gesture steady")
        print("   - Press ESC to cancel\n")
        
        collected = 0
        recording = False
        
        while collected < num_samples:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Failed to read frame")
                break
            
            # Mirror the frame for better UX
            frame = cv2.flip(frame, 1)
            
            # Extract landmarks
            landmarks, hand_detected = self.extractor.extract_landmarks(frame)
            
            # Draw landmarks
            if hand_detected:
                frame = self.extractor.draw_landmarks(frame, color=(0, 255, 0))
                status_color = (0, 255, 0)
                status_text = "Hand Detected"
            else:
                status_color = (0, 0, 255)
                status_text = "No Hand Detected"
            
            # Display status
            cv2.putText(frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Display label and progress
            cv2.putText(frame, f"Gesture: {label}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Collected: {collected}/{num_samples}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Recording indicator
            if recording:
                cv2.putText(frame, "🔴 RECORDING", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save landmark if hand is detected
                if hand_detected and landmarks is not None:
                    csv_row = landmarks_to_csv_row(landmarks, label)
                    with open(self.output_file, 'a') as f:
                        f.write(csv_row + '\n')
                    collected += 1
            else:
                cv2.putText(frame, "Press SPACE to start", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Data Collection', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' '):  # Space to start/pause
                recording = not recording
                if recording:
                    print("🔴 Recording started...")
                else:
                    print("⏸️  Recording paused")
            elif key == 27:  # ESC to cancel
                print("❌ Collection cancelled")
                return False
        
        print(f"✅ Successfully collected {collected} samples for '{label}'")
        return True
    
    def interactive_mode(self):
        """Run interactive data collection session"""
        print("\n" + "="*60)
        print("🎥 SIGN LANGUAGE DATA COLLECTION TOOL")
        print("="*60)
        
        self.start_camera()
        
        # Suggested gestures
        suggested_gestures = [
            "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
            "HELLO", "YES", "NO", "THANKYOU", "ILOVEYOU", "STOP", "NOTHING"
        ]
        
        print("\n📋 Suggested gestures:")
        for i, gesture in enumerate(suggested_gestures, 1):
            print(f"   {i:2d}. {gesture}")
        
        while True:
            print("\n" + "-"*60)
            label = input("\n🏷️  Enter gesture label (or 'quit' to exit): ").strip().upper()
            
            if label.lower() == 'quit':
                print("👋 Exiting...")
                break
            
            if not label:
                print("⚠️  Label cannot be empty")
                continue
            
            # Ask for number of samples
            try:
                num_samples = input(f"📊 Number of samples to collect (default: 500): ").strip()
                num_samples = int(num_samples) if num_samples else 500
            except ValueError:
                print("⚠️  Invalid number, using default (500)")
                num_samples = 500
            
            # Collect data
            success = self.collect_gesture(label, num_samples)
            
            if not success:
                continue_choice = input("\n❓ Continue collecting? (y/n): ").strip().lower()
                if continue_choice != 'y':
                    break
        
        self.cleanup()
    
    def cleanup(self):
        """Release resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()
        print("\n✅ Resources released")
        print(f"📁 Dataset saved to: {os.path.abspath(self.output_file)}")


def main():
    """Main entry point"""
    collector = DataCollector()
    
    try:
        collector.interactive_mode()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        collector.cleanup()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        collector.cleanup()
        raise


if __name__ == "__main__":
    main()
