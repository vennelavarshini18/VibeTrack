import os
import sys
import uuid
import base64
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="🎙️ Track your mood", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mic_input import record_audio
from emotion_predictor import predict_emotion
from plot import show_confidence_chart, show_waveform
from journal import save_to_journal, show_journal

theme = st.get_option("theme.base") or "light"
accent = "#06B6D4" if theme == "light" else "#22d3ee"

st.markdown(f"""
    <style>
      .stApp {{
        background: linear-gradient(135deg, #d0f4ff 0%, #a0e8ff 25%, #80dfff 50%, #a0e8ff 75%, #d0f4ff 100%);
        background-attachment: fixed;
        background-size: cover;
      }}
      .hero-container {{
        background: linear-gradient(135deg, rgba(10,162,212,0.10), rgba(30,64,175,0.10));
        padding: 3rem;
        border-radius: 1.5rem;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        max-width: 2200px;
        margin: auto;
        border: 1px solid rgba(255,255,255,0.12);
        backdrop-filter: blur(6px);
        margin-bottom: 2rem;
      }}
      .hero-image {{
        width: 100%;
        max-width: 300px;
        height: auto;
        margin-bottom: 1.8rem;
        border-radius: 1rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
      }}
      .hero-title {{
        font-size: 6.5em;
        font-weight: 900;
        background: linear-gradient(90deg, #06B6D4, #3B82F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.8rem;
        text-shadow: 0 3px 10px rgba(0,0,0,0.3);
      }}
      .hero-subtitle {{
        font-size: 1.5em;
        font-weight: 500;
        color: {"#000" if theme == "light" else "#fff"};
        line-height: 1.8;
        padding: 0 1rem;
      }}
      .stTabs [data-baseweb="tab"] {{
        background-color: rgba(255, 255, 255, 0.2);
        border-radius: 0.75rem 0.75rem 0 0;
        padding: 0.6rem 1.2rem;
        margin-right: 0.4rem;
        font-weight: 600;
        color: #444;
        transition: all 0.3s ease-in-out;
      }}
      .stTabs [aria-selected="true"] {{
        background-color: {accent};
        color: white;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15);
      }}
      .stTabs [data-baseweb="tab"]:hover {{
        background-color: rgba(0,0,0,0.08);
      }}
    </style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
header_image_path = os.path.join(BASE_DIR, "static", "header_image.png")

if os.path.exists(header_image_path):
    with open(header_image_path, "rb") as img_file:
        encoded_img = base64.b64encode(img_file.read()).decode()
    img_html = f"<img src='data:image/png;base64,{encoded_img}' class='hero-image' alt='VibeTrack Banner'>"
else:
    img_url = "https://raw.githubusercontent.com/vennelavarshini18/YOUR-REPO-NAME/main/static/header_image.png"
    img_html = f"<img src='{img_url}' class='hero-image' alt='VibeTrack Banner'>"

st.markdown(f"""
    <div class="hero-container">
        <div class="hero-title">🎧 VibeTrack</div>
        {img_html}
        <p class="hero-subtitle">
            Step into a world where your voice speaks louder than words.<br>
            VibeTrack is your emotional mirror, powered by deep learning to help you<br>
            understand, reflect, and grow — one emotion at a time.
        </p>
    </div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🎵 Upload Audio", "🎤 Record Mic", "📖 View Journal", "💬 Chatbot"])

with tabs[0]:
    st.markdown("### 🎵 Upload a WAV file to begin:")
    uploaded_file = st.file_uploader("Upload your speech sample", type=["wav"], label_visibility="collapsed")

    if uploaded_file:
        unique_id = str(uuid.uuid4())
        audio_dir = "audio"
        os.makedirs(audio_dir, exist_ok=True)
        audio_path = os.path.join(audio_dir, f"{unique_id}.wav")

        with open(audio_path, "wb") as f:
            f.write(uploaded_file.read())

        st.audio(audio_path, format="audio/wav")
        show_waveform(audio_path)

        emotion, prediction = predict_emotion(audio_path)

        st.success(f"✅ Emotion Detected: {emotion.upper()}")
        st.markdown(
            f"### 🧠 Detected Emotion: <span style='color:{accent}; font-weight:bold;'>{emotion.upper()}</span>",
            unsafe_allow_html=True
        )

        with st.expander("💡 Click for personalized suggestions", expanded=True):
            from utils.suggestions import emotion_reactions
            for idea in emotion_reactions.get(emotion.lower(), ["💪 Stay strong and take care of yourself."]):
                st.markdown(f"- {idea}")

        st.markdown("### 📊 Emotion Probabilities")
        show_confidence_chart(prediction)
        save_to_journal(emotion, prediction)

with tabs[1]:
    st.markdown("### 🎤 Record from Microphone")
    if st.button("Start Recording"):
        file_path = record_audio()
        if file_path:   
            st.audio(file_path, format="audio/wav")
            show_waveform(file_path)

            emotion, prediction = predict_emotion(file_path)

            st.success(f"✅ Emotion Detected: {emotion.upper()}")
            st.markdown(
                f"### 🧠 Detected Emotion: <span style='color:{accent}; font-weight:bold;'>{emotion.upper()}</span>",
                unsafe_allow_html=True
            )

            with st.expander("💡 Click for personalized suggestions", expanded=True):
                from utils.suggestions import emotion_reactions
                for idea in emotion_reactions.get(emotion.lower(), ["💪 Stay strong and take care of yourself."]):
                    st.markdown(f"- {idea}")

            st.markdown("### 📊 Emotion Probabilities")
            show_confidence_chart(prediction)
            save_to_journal(emotion, prediction)
        else:
            st.info("⚠️ Microphone recording is not available here. Please upload a file instead.")

with tabs[2]:
    show_journal()

with tabs[3]:
    st.markdown("### 💬 Emotion-Aware Chatbot")

    if "history" not in st.session_state:
        st.session_state.history = []

    def detect_emotion(text):
        t = text.lower()
        if any(w in t for w in ["sad", "unhappy", "lonely", "depressed"]):
            return "Sad"
        elif any(w in t for w in ["happy", "joy", "great", "excited"]):
            return "Happy"
        elif any(w in t for w in ["angry", "mad", "frustrated"]):
            return "Angry"
        return "Neutral"

    import google.generativeai as genai
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  

    model_name = "gemini-1.5-flash"
    model = genai.GenerativeModel(model_name)

    def get_gemini_reply(user_text, emotion, history):
        history_text = "\n".join([f"{m['role'].capitalize()}: {m['text']}" for m in history])
        prompt = f"""
        You are an empathetic AI assistant.
        The user is currently feeling "{emotion}".
        Respond in a tone that matches this emotion appropriately.
        Use a conversational and human-like style.
        
        Conversation so far:
        {history_text}

        User: {user_text}
        Bot:
        """
        response = model.generate_content(prompt)
        return response.text.strip()

    user_msg = st.text_input("You:", key="chat_input")
    if user_msg:
        emo = detect_emotion(user_msg)
        bot_msg = get_gemini_reply(user_msg, emo, st.session_state.history)
        st.session_state.history.append({"role": "user", "text": user_msg, "emotion": emo})
        st.session_state.history.append({"role": "bot", "text": bot_msg, "emotion": emo})

    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['text']}")
        else:
            st.markdown(f"**Bot:** {msg['text']}")

    if st.session_state.history:
        emos = [m["emotion"] for m in st.session_state.history if m["role"] == "user"]
        counts = {e: emos.count(e) for e in set(emos)}
        fig, ax = plt.subplots()
        ax.bar(counts.keys(), counts.values(), color=['skyblue', 'lightgreen', 'salmon', 'gray'])
        ax.set_ylabel("Frequency")
        ax.set_title("Detected Emotions in Conversation")
        st.pyplot(fig)
