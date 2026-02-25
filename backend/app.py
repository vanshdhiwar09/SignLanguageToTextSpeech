"""
FastAPI + SocketIO Server for Real-Time Sign Language Translation
"""

import asyncio
import json
import base64
import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio
from tensorflow import keras
import pyttsx3
import threading
from utils.landmarks import LandmarkExtractor
from utils.mediator import PredictionMediator
import os


# Initialize FastAPI
app = FastAPI(title="Sign Language Translator API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize SocketIO
sio = socketio.AsyncServer(
    async_mode='asgi',
    cors_allowed_origins='*',
    logger=True,
    engineio_logger=False
)

# Combine FastAPI and SocketIO
socket_app = socketio.ASGIApp(sio, app)

# Global instances
landmark_extractor = None
prediction_mediator = None
model = None
class_labels = []
tts_engine = None


def load_model_and_metadata():
    """Load the trained model and metadata"""
    global model, class_labels
    
    model_path = "model.h5"
    metadata_path = "model_metadata.json"
    
    if not os.path.exists(model_path):
        print(f"⚠️  Model not found at {model_path}")
        print("   Please train the model first using train_model.py")
        return False
    
    # Load model
    model = keras.models.load_model(model_path)
    print(f"✅ Model loaded from {model_path}")
    
    # Load metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            class_labels = metadata.get('classes', [])
        print(f"✅ Loaded {len(class_labels)} class labels")
    else:
        print("⚠️  Metadata not found, using default labels")
        class_labels = [f"CLASS_{i}" for i in range(model.output_shape[-1])]
    
    return True


def initialize_tts():
    """TTS available - engines created on demand"""
    print("✅ TTS ready (creates fresh engine per request for reliability)")


@app.on_event("startup")
async def startup_event():
    """Initialize components on startup"""
    global landmark_extractor, prediction_mediator
    
    print("\n" + "="*60)
    print("🚀 STARTING SIGN LANGUAGE TRANSLATOR SERVER")
    print("="*60)
    
    # Load model
    if not load_model_and_metadata():
        print("❌ Server cannot start without a trained model")
        return
    
    # Initialize landmark extractor
    landmark_extractor = LandmarkExtractor()
    print("✅ Landmark extractor initialized")
    
    # Initialize prediction mediator
    prediction_mediator = PredictionMediator(
        buffer_size=15,
        stability_threshold=12,
        confidence_threshold=0.85,
        hold_duration=1.5
    )
    print("✅ Prediction mediator initialized")
    
    # Initialize TTS
    initialize_tts()
    
    print("\n✅ Server ready!")
    print("="*60 + "\n")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Sign Language Translator",
        "model_loaded": model is not None,
        "num_classes": len(class_labels)
    }


@app.get("/classes")
async def get_classes():
    """Get available gesture classes"""
    return {
        "classes": class_labels,
        "count": len(class_labels)
    }


@sio.event
async def connect(sid, environ):
    """Handle client connection"""
    print(f"🔌 Client connected: {sid}")
    await sio.emit('connection_status', {'status': 'connected', 'sid': sid}, room=sid)


@sio.event
async def disconnect(sid):
    """Handle client disconnection"""
    print(f"🔌 Client disconnected: {sid}")


@sio.event
async def video_frame(sid, data):
    """
    Process video frame from client
    
    Expected data format:
    {
        'frame': base64_encoded_image
    }
    """
    try:
        # Decode base64 image
        img_data = base64.b64decode(data['frame'].split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Extract landmarks
        landmarks, hand_detected = landmark_extractor.extract_landmarks(frame)
        
        if hand_detected and landmarks is not None:
            # Make prediction
            landmarks_reshaped = landmarks.reshape(1, -1)
            predictions = model.predict(landmarks_reshaped, verbose=0)[0]
            
            # Get top prediction
            top_idx = np.argmax(predictions)
            top_label = class_labels[top_idx]
            top_confidence = float(predictions[top_idx])
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = [
                {
                    'label': class_labels[idx],
                    'confidence': float(predictions[idx])
                }
                for idx in top_3_indices
            ]
            
            # Add to mediator for stability
            result = prediction_mediator.add_prediction(top_label, top_confidence)
            
            # Send response
            await sio.emit('prediction', {
                'hand_detected': True,
                'raw_prediction': top_label,
                'raw_confidence': top_confidence,
                'stable_prediction': result['stable_prediction'],
                'stable_confidence': result['confidence'],
                'sentence': result['sentence'],
                'top_3': top_3_predictions,
                'buffer_status': result['buffer_status']
            }, room=sid)
        else:
            # No hand detected - clear buffer immediately for instant reset
            prediction_mediator.clear_buffer()
            await sio.emit('prediction', {
                'hand_detected': False,
                'stable_prediction': 'NOTHING',
                'sentence': prediction_mediator.get_sentence()
            }, room=sid)
    
    except Exception as e:
        print(f"❌ Error processing frame: {e}")
        await sio.emit('error', {'message': str(e)}, room=sid)


@sio.event
async def clear_text(sid, data):
    """Clear the current sentence"""
    prediction_mediator.clear_sentence()
    await sio.emit('text_cleared', {'sentence': ''}, room=sid)


@sio.event
async def speak_text(sid, data):
    """Convert text to speech (fire-and-forget for guaranteed execution)"""
    text = data.get('text', '')
    
    if not text:
        await sio.emit('tts_error', {'message': 'No text to speak'}, room=sid)
        return
    
    try:
        # Send immediate acknowledgment
        await sio.emit('tts_started', {'text': text}, room=sid)
        
        # Fire-and-forget: Start thread and return immediately
        def speak():
            try:
                engine = pyttsx3.init()
                engine.setProperty('rate', 250)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                print(f"🔊 TTS completed: {text}")
            except Exception as e:
                print(f"❌ TTS thread error: {e}")
        
        # Start thread (daemon=True means it won't block shutdown)
        thread = threading.Thread(target=speak, daemon=True)
        thread.start()
        
        # Return immediately - don't wait for speech to finish
        await sio.emit('tts_complete', {'text': text}, room=sid)
        print(f"🔊 TTS thread started for: {text}")
    except Exception as e:
        print(f"❌ TTS error: {e}")
        await sio.emit('tts_error', {'message': str(e)}, room=sid)


@sio.event
async def reset_session(sid, data):
    """Reset the prediction session"""
    prediction_mediator.reset()
    await sio.emit('session_reset', {'status': 'reset'}, room=sid)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        socket_app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
