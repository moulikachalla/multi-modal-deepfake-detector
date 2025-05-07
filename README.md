# multi-modal-deepfake-detector

This repository contains a multi-modal deepfake detection system that classifies **images**, **videos**, and **audio** as real or fake using deep learning. Built using **custom CNN**, **LSTM**, and **ANN** models, with a **Streamlit** web interface for real-time media classification.

## 🚀 Features

- 🖼️ Image Detection using **custom CNN + ANN** – 83.5% validation accuracy  
- 🎞️ Video Detection using **custom CNN + LSTM** – 80.9% validation accuracy  
- 🔊 Audio Detection using **ANN** with **MFCC** and **spectral contrast features** via **Librosa** – 98.2% validation accuracy  
- 🌐 Real-time media classification via **Streamlit** web application

## 🧠 Model Architectures

- **Images**: Custom-built CNN → Flatten → ANN for binary classification  
- **Videos**: Frame-wise Custom CNN → LSTM for temporal learning  
- **Audio**: MFCC + spectral features → ANN classifier

## 📁 Dataset Sources

- 📸 [Deepfake and Real Images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)  
- 🎬 [Deepfake Video Detection Challenge](https://www.kaggle.com/competitions/deepfake-detection-challenge)  
- 🎧 [Fake or Real Audio Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)

## ⚙️ Tech Stack

- Python 3.10.12  
- TensorFlow 2.17.1, Keras 3.5.0  
- Librosa 0.10.0, MoviePy 1.0.3  
- OpenCV, Pillow  
- Streamlit 1.44.1  

## 🖥️ Web App Functionality

- Upload media (image, video, audio)  
- Get real-time classification result with confidence score  
- Visual interface for previewing predictions

## 📈 Results Summary

| Modality | Model            | Validation Accuracy |
|----------|------------------|---------------------|
| Image    | Custom CNN + ANN | 83.5%               |
| Video    | Custom CNN + LSTM| 80.9%               |
| Audio    | ANN + Librosa    | 98.2%               |

## 🔮 Future Enhancements

- Real-time stream-based detection (e.g., live webcam/video)  
- Improve temporal accuracy using advanced sequence models  
- Multi-language and multi-accent audio detection

## 👩‍💻 Author

**Moulika Challa**  
Master’s Student, Computer Science  
California State University, Sacramento


