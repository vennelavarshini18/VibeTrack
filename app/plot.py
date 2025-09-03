import matplotlib.pyplot as plt
import plotly.graph_objects as go
import streamlit as st
import librosa
import librosa.display
import pandas as pd
import numpy as np
import wave

def show_waveform(file_path):
    with wave.open(file_path, 'rb') as wav_file:
        n_frames = wav_file.getnframes()
        framerate = wav_file.getframerate()
        signal = wav_file.readframes(n_frames)
        waveform = np.frombuffer(signal, dtype=np.int16)
        time = np.linspace(0, len(waveform) / framerate, num=len(waveform))

    fig, ax = plt.subplots(figsize=(6, 3))  
    ax.plot(time, waveform, linewidth=0.5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_title("Audio Waveform")
    ax.grid(True)

    left, center, right = st.columns([1, 2.5, 1])
    with center:
        st.markdown(
            """
            <div style="background-color:#1e1e1e; padding:1rem; border-radius:12px;
                        box-shadow: 0 4px 10px rgba(0,0,0,0.4); margin-top: 1rem;">
            """,
            unsafe_allow_html=True
        )
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

emotion_colors = {
    "angry": "#FF4C4C",
    "disgust": "#6B8E23",
    "fear": "#FFA07A",
    "happy": "#00CC96",
    "neutral": "#A9A9A9",
    "sad": "#1E90FF",
    "surprise": "#FFD700"
}

def show_confidence_chart(predictions_dict):
    df = pd.DataFrame({
        "Emotion": list(predictions_dict.keys()),
        "Probability": list(predictions_dict.values())
    }).sort_values(by="Probability", ascending=False)

    colors = [emotion_colors.get(emotion, "#888888") for emotion in df["Emotion"]]

    fig = go.Figure(
        data=[
            go.Bar(
                x=df["Emotion"],
                y=df["Probability"],
                marker_color=colors,
                text=[f"{p*100:.1f}%" for p in df["Probability"]],
                textposition="auto",
                hovertemplate='%{x}: <b>%{text}</b><extra></extra>',
                marker_line_width=1.2,
            )
        ]
    )

    fig.update_layout(
        title="📊 <b>Emotion Probabilities</b>",
        xaxis_title="Emotion",
        yaxis_title="Probability",
        template="plotly_white",
        plot_bgcolor="#f5f8ff",
        paper_bgcolor="#f5f8ff",
        font=dict(family="Segoe UI", size=14),
        margin=dict(l=30, r=30, t=60, b=30),
        height=450
    )

    st.plotly_chart(fig, use_container_width=True)