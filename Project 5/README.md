# 😊 Real-Time Emotion Detection System

A real-time facial emotion detection system built with OpenCV and Keras that identifies 7 human emotions directly from your webcam feed — no cloud API or DeepFace dependency required.

---

## 📸 Demo

> The system captures your webcam feed, detects faces using Haar Cascade, and overlays the predicted emotion label in real time.

---

## 🧠 How It Works

1. **Face Detection** — OpenCV's Haar Cascade classifier (`haarcascade_frontalface_default.xml`) locates faces in each video frame.
2. **Preprocessing** — The detected face region is converted to grayscale, resized to `64×64`, and normalized to `[0, 1]`.
3. **Emotion Classification** — A pre-trained Keras CNN model (`emotion_model.hdf5`) predicts one of 7 emotion classes.
4. **Visualization** — A bounding box and emotion label are drawn over the detected face in real time.

---

## 🎭 Supported Emotions

| Label     | Description              |
|-----------|--------------------------|
| Angry     | Anger / Frustration      |
| Disgust   | Disgust                  |
| Fear      | Fear / Anxiety           |
| Happy     | Happiness / Joy          |
| Sad       | Sadness                  |
| Surprise  | Surprise / Shock         |
| Neutral   | No strong emotion        |

---

## 🗂️ Project Structure

```
emotion-detection/
│
├── emotion_detection.py               # Main script — run this to start the system
├── emotion_model.hdf5                 # Pre-trained Keras CNN model
├── haarcascade_frontalface_default.xml  # OpenCV face detector
└── README.md
```

---

## ⚙️ Requirements

- Python 3.7+
- OpenCV
- NumPy
- TensorFlow / Keras

Install all dependencies:

```bash
pip install opencv-python numpy tensorflow
```

> **macOS users:** The script uses `cv2.CAP_AVFOUNDATION` for webcam access, which is the recommended backend on macOS. No extra configuration needed.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/emotion-detection.git
cd emotion-detection
```

### 2. Install dependencies

```bash
pip install opencv-python numpy tensorflow
```

### 3. Run the detection script

```bash
python emotion_detection.py
```

### 4. Quit

Press **`q`** to exit the webcam window.

---

## 🔍 Code Overview

```python
# Load models
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
emotion_model = load_model("emotion_model.hdf5", compile=False)

# Per-frame pipeline
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    roi = cv2.resize(gray[y:y+h, x:x+w], (64, 64)) / 255.0
    prediction = emotion_model.predict(roi.reshape(1, 64, 64, 1))
    emotion = emotion_labels[np.argmax(prediction)]
```

---

## 🛠️ Troubleshooting

| Issue | Fix |
|-------|-----|
| Webcam not opening | Check camera permissions; try changing `VideoCapture(0)` index to `1` |
| Low detection accuracy | Ensure good lighting and face the camera directly |
| `model not found` error | Confirm `emotion_model.hdf5` is in the same directory as the script |
| Slow performance | Reduce frame resolution or run on a machine with GPU support |

---

## 📦 Model Details

- **Input:** Grayscale image, shape `(1, 64, 64, 1)`
- **Output:** Softmax probabilities over 7 emotion classes
- **Architecture:** Convolutional Neural Network (CNN)
- **Format:** Keras HDF5 (`.hdf5`)

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests for improvements such as:

- Multi-face tracking
- Emotion history / analytics
- Support for video file input
- GUI interface

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/) for face detection and video capture
- [Keras / TensorFlow](https://keras.io/) for the deep learning model
- FER-2013 dataset (commonly used for training emotion classifiers)
