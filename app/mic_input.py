import os
import datetime
import numpy as np
import streamlit as st

try:
    import sounddevice as sd
    import soundfile as sf
    MIC_AVAILABLE = True
except Exception:
    MIC_AVAILABLE = False
    st.warning("🎙️ Microphone recording is not allowed in this environment.")


def record_audio(duration: int = 4, fs: int = 16000, output_dir: str = "audio") -> str | None:
    if not MIC_AVAILABLE:
        st.info("⚠️ Microphone recording is not allowed.")
        return None

    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(
        output_dir, f"mic_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.wav"
    )

    try:
        st.toast("Recording started...", icon="🎙️")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
        sd.wait()

        if recording is None or recording.size == 0:
            st.info("⚠️ Microphone recording is not allowed.")
            return None

        recording = recording.squeeze()
        max_val = np.max(np.abs(recording))
        if max_val > 0:
            recording = recording / max_val

        sf.write(filename, recording, fs)
        st.toast("✅ Recording complete!", icon="✅")
        return filename

    except Exception:
        st.info("⚠️ Microphone recording is not allowed.")
        return None
