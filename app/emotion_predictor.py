import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2" 
import numpy as np
import librosa
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best_model.h5")
LABEL_PATH = os.path.join(BASE_DIR, "..", "model", "label_classes.npy")

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(2376, 1)),  
    tf.keras.layers.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same'),

    tf.keras.layers.Conv1D(512, kernel_size=5, strides=1, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(256, kernel_size=5, strides=1, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same'),

    tf.keras.layers.Conv1D(256, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=5, strides=2, padding='same'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Conv1D(128, kernel_size=3, strides=1, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling1D(pool_size=3, strides=2, padding='same'),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(6, activation='softmax')
])

model.load_weights(MODEL_PATH)
label_classes = np.load(LABEL_PATH, allow_pickle=True)

def zcr(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length))

def rmse(data, frame_length=2048, hop_length=512):
    return np.squeeze(librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length))

def mfcc(data, sr, flatten=True):
    mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=30)
    return np.ravel(mfccs.T) if flatten else np.squeeze(mfccs.T)

def extract_features(data, sr=22050, max_len=2376):
    result = np.hstack((
        zcr(data),
        rmse(data),
        mfcc(data, sr)
    ))

    if len(result) > max_len:
        result = result[:max_len]
    elif len(result) < max_len:
        result = np.pad(result, (0, max_len - len(result)))

    return result

def get_features(path, duration=2.5, offset=0.6):
    data, sr = librosa.load(path, duration=duration, offset=offset)
    return extract_features(data, sr)

def predict_emotion(file_path):
    features = get_features(file_path)
    features = np.expand_dims(features, axis=0)  
    features = np.expand_dims(features, axis=2) 

    prediction = model.predict(features, verbose=0)[0]
    emotion = label_classes[np.argmax(prediction)]

    print(f"[INFO] Prediction complete for: {file_path}")
    print(f"[INFO] Detected Emotion: {emotion}")

    return emotion, dict(zip(label_classes, prediction.tolist()))
