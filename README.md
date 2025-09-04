# VibeTrack: Track Your Mood

**VibeTrack** is a deep learning–powered web application that detects your **inner emotional state** from your **voice** — either through uploading `.wav` files or recording in real-time — and also features an **emotion-aware chatbot** along with a **mood-tracking journal** for **personalized mental wellness support**.

Unlike traditional NLP-based sentiment models, **VibeTrack is sentence-agnostic** — it listens to **tone, pitch, rhythm, and frequency** patterns rather than the words you say.

---

## Features

- **Real-time microphone emotion detection**
- **.wav file upload support**
- **Custom CNN architecture trained on MFCC features**
- **Emotion prediction with confidence scores**
- **Mood Journal to track your emotional patterns**
- **Personalized suggestions based on detected emotion**
- **Clean UI with waveform and probability visualizations**
- **Emotion-aware chatbot with conversation memory, Gemini-powered tone adaptation, and visual insights**

---

## How It Works

VibeTrack uses **MFCC-based preprocessing** to extract important audio features like:

- **Pitch modulation**
- **Spectral energy**
- **Voice intensity**
- **Mel-frequency cepstral coefficients (MFCCs)**

These are passed into a **Convolutional Neural Network (CNN)** trained on a diverse emotional speech dataset. The model achieved **99.9% validation accuracy** during training.

It detects the following emotions:

> `Happy, Sad, Angry, Fearful, Neutral, Disgust`

The system not only visualizes prediction probabilities with dynamic charts but also offers personalized suggestions—like **uplifting activities or calming techniques**—tailored to the user’s current mood. 

All interactions are securely logged into a **built-in mood journal**, enabling users to **track emotional patterns** over time.

The chatbot also **keeps conversation memory** so it remembers previous interactions in the ongoing chat and **understands emotional tone** from the conversation history to adapt its replies (e.g., more compassionate if sadness is detected).

---

## Live Demo

- **Find here:**  [Deployed on Streamlit Cloud](https://trackyourmood.streamlit.app/) 
- **Demo video:** [Watch on Google Drive](https://drive.google.com/file/d/1ow1KIWIiqMHQIy5jEuyA1xRvyxLcmsvJ/view?usp=sharing)

---

## Why Use VibeTrack?

> "We don’t always say what we feel — but our voice does."

VibeTrack acts like a mirror to your emotions. Whether you're working on **mental wellness, emotional journaling, or voice-based AI systems**, **VibeTrack helps you understand how your voice reflects your feelings**.

---

## Tech Stack

- **Python 3.10+**
- **Streamlit**
- **TensorFlow / Keras**
- **Librosa** for audio feature extraction (MFCCs)
- **SoundDevice** for microphone recording
- **Matplotlib / Seaborn / Plotly** for visualizations

---

## Model Performance

- **Accuracy:** 99.9% on validation set  
- **Dataset:** Cleaned a large dataset with wide variety of audio files  
- **Preprocessing:** MFCCs with zero-padding and time-frequency normalization  
- **Architecture:** Custom 1D CNN with dropout and batch normalization layers  

---
