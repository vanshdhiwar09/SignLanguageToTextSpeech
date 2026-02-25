# 🤖 Sign Language Translator - Real-Time AI System

A **Final Year B.Sc. Computer Science Project** that translates American Sign Language (ASL) gestures into text and speech in real-time using MediaPipe landmarks and a custom MLP neural network.

## 🎯 Features

- ✅ **Real-time gesture recognition** using webcam
- ✅ **32+ gesture classes** (A-Z alphabet + common words)
- ✅ **Landmark-based MLP model** (no CNN/image classification)
- ✅ **WebSocket architecture** for low-latency streaming
- ✅ **Anti-flicker prediction smoothing** with voting buffer
- ✅ **Text-to-Speech (TTS)** integration
- ✅ **Modern cyberpunk UI** with dark mode
- ✅ **Custom data collection tool** for building your own dataset

---

## 📂 Project Structure

```
sign-language-translator/
├── backend/
│   ├── app.py                 # FastAPI + SocketIO server
│   ├── collect_data.py        # Data collection tool
│   ├── train_model.py         # Model training script
│   ├── requirements.txt       # Python dependencies
│   ├── utils/
│   │   ├── landmarks.py       # MediaPipe landmark extraction
│   │   └── mediator.py        # Prediction smoothing logic
│   ├── data/
│   │   └── gesture_data.csv   # Your collected dataset (created by collect_data.py)
│   ├── model.h5               # Trained model (created by train_model.py)
│   └── model_metadata.json    # Class labels and metadata
│
└── frontend/
    ├── src/
    │   ├── components/
    │   │   ├── CameraFeed.jsx
    │   │   ├── ConfidenceChart.jsx
    │   │   └── TextBox.jsx
    │   ├── App.jsx
    │   ├── socket.js
    │   └── index.css
    ├── package.json
    └── tailwind.config.js
```

---

## 🚀 Quick Start Guide

### **Prerequisites**

- **Python 3.11+**
- **Node.js 18+** and npm
- **Webcam** (for data collection and real-time translation)
- **Windows/Linux/Mac** (tested on Windows)

---

## 📦 Installation

### **1. Backend Setup**

```powershell
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### **2. Frontend Setup**

```powershell
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

---

## 🎬 Usage Workflow

### **Step 1: Collect Your Dataset**

Since this project uses a **custom dataset**, you need to record your own sign language gestures.

```powershell
cd backend
python collect_data.py
```

**Instructions:**
1. The webcam will open
2. Enter a gesture label (e.g., `A`, `HELLO`, `THANKYOU`)
3. Press **SPACE** to start recording
4. Hold the gesture steady (the system will capture 500 frames by default)
5. Repeat for all gestures you want to recognize

**Recommended gestures:**
- **Alphabet:** A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z
- **Words:** HELLO, YES, NO, THANKYOU, ILOVEYOU, STOP
- **Neutral:** NOTHING (no hand or resting position)

**Tips:**
- Collect **500-1000 samples per gesture** for best accuracy
- Vary hand position, rotation, and distance from camera
- Use good lighting
- The data is saved to `backend/data/gesture_data.csv`

---

### **Step 2: Train the Model**

Once you have collected data for all gestures:

```powershell
cd backend
python train_model.py
```

**What happens:**
- Loads data from `data/gesture_data.csv`
- Displays class distribution
- Trains an MLP model with:
  - 2 hidden layers (128 neurons each)
  - Dropout for regularization
  - Class balancing for imbalanced data
- Saves:
  - `model.h5` (trained model)
  - `model_metadata.json` (class labels)
  - `training_history.png` (accuracy/loss plots)

**Expected output:**
```
✅ Loaded 15000 samples
📊 Class Distribution:
   A: 500 samples
   B: 500 samples
   ...
✅ Training completed!
   Test Accuracy: 95.23%
```

---

### **Step 3: Run the Application**

#### **Start the Backend Server**

```powershell
cd backend
python app.py
```

The server will start on `http://localhost:8000`

#### **Start the Frontend**

In a **new terminal**:

```powershell
cd frontend
npm run dev
```

The frontend will start on `http://localhost:3000`

---

### **Step 4: Use the Translator**

1. Open `http://localhost:3000` in your browser
2. **Allow camera permissions** when prompted
3. **Position your hand** in the camera view
4. **Make a sign** and hold it for ~1.5 seconds
5. The gesture will be:
   - Detected and shown in "Current Gesture"
   - Added to the sentence after holding
   - Spoken aloud (if Auto-Speak is enabled)

**Controls:**
- **Auto-Speak Toggle:** Automatically speaks new text
- **Speak Text:** Manually trigger TTS
- **Clear Text:** Clear the current sentence
- **Reset:** Reset all stats and predictions

---

## 🧠 How It Works

### **1. Landmark Extraction**
- Uses **MediaPipe Hands** to detect 21 hand landmarks (x, y, z)
- Normalizes landmarks:
  - **Wrist-relative:** Subtracts wrist position from all points
  - **Scale-invariant:** Divides by max distance (handles zoom/depth)
- Outputs **63 features** (21 landmarks × 3 coordinates)

### **2. Model Architecture (MLP)**
```
Input (63) → Dense(128, ReLU) → Dropout(0.2) →
Dense(128, ReLU) → Dropout(0.2) →
Dense(64, ReLU) →
Output(Softmax, Num_Classes)
```

### **3. Prediction Smoothing**
To prevent flickering:
- **Buffer:** Keeps last 15 predictions
- **Voting:** Only outputs gesture if it appears in ≥12 of last 15 frames
- **Confidence threshold:** Discards predictions < 85%
- **Hold duration:** Gesture must be stable for 1.5s before adding to sentence

### **4. WebSocket Communication**
- **Client** captures video frames (10 FPS)
- Sends base64-encoded frames to **server**
- **Server** processes frame → extracts landmarks → predicts → sends result
- **Client** updates UI in real-time

---

## 🎨 UI Features

- **Cyberpunk/Dark Theme** with neon accents
- **Live camera feed** with status indicators
- **Real-time confidence chart** (top 3 predictions)
- **Animated text box** with typing cursor
- **Statistics dashboard** (frames processed, avg confidence, word count)
- **Responsive design** for desktop and tablet

---

## 🛠️ Customization

### **Add More Gestures**

1. Run `collect_data.py` and record new gestures
2. Re-train the model with `train_model.py`
3. Restart the backend server

### **Adjust Prediction Sensitivity**

Edit `backend/utils/mediator.py`:

```python
prediction_mediator = PredictionMediator(
    buffer_size=15,           # Increase for more stability
    stability_threshold=12,   # Increase for stricter voting
    confidence_threshold=0.85, # Increase to reject low-confidence predictions
    hold_duration=1.5         # Increase to require longer holds
)
```

### **Change Model Architecture**

Edit `backend/train_model.py` in the `create_model()` method.

---

## 📊 Performance Tips

### **For Better Accuracy:**
- Collect **more samples per gesture** (1000+)
- Ensure **balanced dataset** (similar samples per class)
- Use **good lighting** and **plain background**
- Train for **more epochs** (edit `train_model.py`)

### **For Faster Inference:**
- Reduce frame rate in `CameraFeed.jsx` (change interval from 100ms to 200ms)
- Use smaller model (reduce hidden layer sizes)

---

## 🐛 Troubleshooting

### **Camera not working**
- Grant camera permissions in browser
- Check if another app is using the camera
- Try a different browser (Chrome recommended)

### **Model not found error**
- Make sure you've run `train_model.py` first
- Check that `model.h5` exists in `backend/` directory

### **Low accuracy**
- Collect more data (especially for confused classes)
- Check class distribution (should be balanced)
- Increase training epochs
- Verify landmarks are being extracted correctly

### **WebSocket connection failed**
- Ensure backend is running on port 8000
- Check firewall settings
- Verify `SOCKET_URL` in `frontend/src/socket.js`

### **TTS not working**
- TTS uses `pyttsx3` which requires system TTS engines
- On Windows: Should work out of the box
- On Linux: Install `espeak` (`sudo apt-get install espeak`)
- On Mac: Should work with built-in TTS

---

## 📝 Technical Specifications

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | FastAPI + Python-SocketIO |
| **ML Framework** | TensorFlow/Keras |
| **Hand Detection** | MediaPipe Hands |
| **Model Type** | Feedforward Neural Network (MLP) |
| **Input Features** | 63 (21 landmarks × 3 coords) |
| **Frontend Framework** | React + Vite |
| **Styling** | Tailwind CSS |
| **Charts** | Recharts |
| **Real-time Communication** | WebSockets (Socket.IO) |
| **TTS** | pyttsx3 |

---

## 🎓 Academic Context

This project demonstrates:
- **Computer Vision:** MediaPipe landmark detection
- **Machine Learning:** Custom MLP classifier with class balancing
- **Real-time Systems:** WebSocket streaming architecture
- **Data Engineering:** Custom data collection pipeline
- **Full-stack Development:** React frontend + FastAPI backend
- **UX Design:** Modern, accessible interface

---

## 📄 License

This is an academic project for educational purposes.

---

## 🙏 Acknowledgments

- **MediaPipe** for hand landmark detection
- **TensorFlow** for ML framework
- **FastAPI** for high-performance backend
- **React** and **Tailwind CSS** for modern UI

---

## 📧 Support

For issues or questions:
1. Check the **Troubleshooting** section
2. Review the code comments
3. Verify all dependencies are installed correctly

---

**Built with ❤️ for Final Year B.Sc. CS Project**
