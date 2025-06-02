# DSRTER
Dual Stream Real Time Facial Emotion Recognition Leveraging Facial Features and Speech Embeddings 

Tech Stack 
Python, VS Code, Google Collab, Web Cam, Mic

Project Overview

The need for intelligent human-computer interaction systems has grown rapidly, especially those capable of understanding human emotions. This project proposes a dual-stream real-time emotion recognition system that classifies seven basic emotions—Happy, Sad, Angry, Neutral, Surprise, Disgust, and Fear—by combining facial expression analysis and speech-based emotion detection.
In the Real-Time Facial Emotion Recognition (RTFER) stream, the FER-2013 dataset is used. MediaPipe performs real-time face detection and tracking, while features are extracted using a hybrid model combining MobileNetV2 and a custom CNN, ensuring a balance between accuracy and speed.
The second stream, Speech-Independent Emotion Recognition (SIER), uses the CREMA-D and RAVDESS datasets. Emotional features are extracted using OpenSMILE (for acoustic features) and Wav2Vec 2.0 (for contextual speech embeddings). These features are then classified by a 1D-CNN trained to detect emotional states from speech.
The system is developed in Python, trained using Google Colab with GPU acceleration. OpenCV and MediaPipe handle real-time video input, while Librosa and SoundDevice process audio input. System performance is evaluated using metrics such as accuracy, precision, F1-score, confusion matrix, and z-score normalization.
In real-time deployment, the system displays the user's face in a bounding box with the predicted facial emotion above and speech emotion below, offering intuitive feedback. This integrated approach demonstrates the potential of multimodal deep learning for building emotionally intelligent AI systems.



In the reposotory I have uploaded some files that are as follows:
1) DSRTER_uisng_Facial_&_Speech  --->  Training File
2) DSRTER Main Code ---> Testing or Execution File (Realtime)
3) DSRTER on video ---> execcution on a video and audio input
4) RTFER_model.keras and SIER_model.keras ---> Training weights model file (can be directly used in testing code and executed)
