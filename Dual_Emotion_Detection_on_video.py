# ------------------------------------------------------------
# Import Required Libraries
# ------------------------------------------------------------
import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import queue
import torch
import librosa
import time
from scipy.special import softmax
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import opensmile
import mediapipe as mp

# ------------------------------------------------------------
# Load Trained Models
# ------------------------------------------------------------
rtfer_model = tf.keras.models.load_model("final_rtfer_custom_cnn.keras") 
sier_model = tf.keras.models.load_model("best_model_sier_50epochs.keras")

processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# ------------------------------------------------------------
# OpenSMILE Feature Extractor
# ------------------------------------------------------------
smile = opensmile.Smile(
    feature_set=opensmile.FeatureSet.eGeMAPSv02,
    feature_level=opensmile.FeatureLevel.Functionals
)

# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
def denoise_audio(audio, sr):
    return librosa.effects.preemphasis(audio)

# ------------------------------------------------------------
# Facial Emotion Detection
# ------------------------------------------------------------

face_detector = mp.solutions.face_detection.FaceDetection(0.7)
'''
def detect_face_emotion(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            x, y = max(0, x), max(0, y)
            cropped = frame[y:y+height, x:x+width]

            resized = cv2.resize(cropped, (48, 48))
            normalized = resized / 255.0
            input_face = np.expand_dims(normalized, axis=0)

            preds = rtfer_model.predict(input_face, verbose=0)
            return np.argmax(preds), softmax(preds[0]), (x, y, width, height)

    return None, None, None
'''
# code for custom CNN framework 
def detect_face_emotion(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)

    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = frame.shape
            x, y = int(bbox.xmin * w), int(bbox.ymin * h)
            width, height = int(bbox.width * w), int(bbox.height * h)
            x, y = max(0, x), max(0, y)
            cropped = frame[y:y+height, x:x+width]

            resized = cv2.resize(cropped, (48, 48))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            normalized = rgb / 255.0
            input_face = np.expand_dims(normalized, axis=0)

            preds = rtfer_model.predict(input_face, verbose=0)
            return np.argmax(preds), softmax(preds[0]), (x, y, width, height)

    return None, None, None


# ------------------------------------------------------------
# Speech Emotion Detection (Wav2Vec + SIER)
# ------------------------------------------------------------
def get_speech_emotion(audio):
    audio = denoise_audio(audio, 16000)
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        wav_features = wav2vec_model(**inputs).last_hidden_state.mean(dim=1).numpy()
    smile_feats = smile.process_signal(audio, 16000).values.flatten()
    combined = np.concatenate([wav_features.flatten(), smile_feats]).reshape(1, -1, 1)
    preds = sier_model.predict(combined, verbose=0)
    return np.argmax(preds), softmax(preds[0])

# ------------------------------------------------------------
# Emotion Labels (6 Class)
# ------------------------------------------------------------
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad']

def get_label_from_index(idx):
    return emotion_labels[idx] if idx is not None and idx < len(emotion_labels) else "Unknown"

# ------------------------------------------------------------
# Late Fusion
# ------------------------------------------------------------
def soft_voting(facial_probs, speech_probs):
    if facial_probs is None:
        facial_probs = np.zeros(6)
    if speech_probs is None:
        speech_probs = np.zeros(6)
    combined = (facial_probs + speech_probs) / 2
    return np.argmax(combined), combined

# ------------------------------------------------------------
# Draw Utility
# ------------------------------------------------------------
def draw_text_with_background(img, text, pos, font, scale, text_color, bg_color, thickness=2):
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    x, y = pos
    cv2.rectangle(img, (x, y - size[1] - 10), (x + size[0] + 10, y + 5), bg_color, -1)
    cv2.putText(img, text, (x + 5, y - 5), font, scale, text_color, thickness)

# ------------------------------------------------------------
# Real-Time Dual Emotion Detection
# ------------------------------------------------------------
def run_emotion_detection_from_video(video_path):
    print(f"Processing video file: {video_path}")
    cap = cv2.VideoCapture(video_path)

    # Extract audio using librosa
    import moviepy.editor as mpedit
    temp_audio_path = "temp_audio.wav"
    clip = mpedit.VideoFileClip(video_path)
    clip.audio.write_audiofile(temp_audio_path, fps=16000)

    y_audio, sr = librosa.load(temp_audio_path, sr=16000)
    total_audio_samples = len(y_audio)

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    audio_samples_per_frame = int(sr / frame_rate)
    current_audio_index = 0

    while True:
        ret, frame = cap.read()
        if not ret or current_audio_index + audio_samples_per_frame >= total_audio_samples:
            break

        # Extract audio chunk for the frame
        audio_chunk = y_audio[current_audio_index:current_audio_index + audio_samples_per_frame]
        current_audio_index += audio_samples_per_frame

        # --- Facial Emotion ---
        facial_idx, facial_probs, box = detect_face_emotion(frame)

        # --- Speech Emotion ---
        try:
            speech_idx, speech_probs = get_speech_emotion(audio_chunk)
        except Exception as e:
            print("Audio error:", e)
            speech_idx, speech_probs = None, None

        # --- Late Fusion ---
        combined_idx, _ = soft_voting(facial_probs, speech_probs)

        # --- Draw results ---
        if box:
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            draw_text_with_background(frame, f"Face: {get_label_from_index(facial_idx)}",
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))
            draw_text_with_background(frame, f"Speech: {get_label_from_index(speech_idx)}",
                                      (x, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), (0, 0, 0))

        draw_text_with_background(frame, f"Combined: {get_label_from_index(combined_idx)}",
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), (0, 0, 255))

        cv2.imshow("Video Emotion Detection", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# ------------------------------------------------------------
# Run
# ------------------------------------------------------------
if __name__ == "__main__":
    run_emotion_detection_from_video("WIN_20250526_18_42_26_Pro.mp4")







