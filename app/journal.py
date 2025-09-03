import pandas as pd
import os
import datetime
import streamlit as st
from pandas.errors import EmptyDataError

JOURNAL_PATH = "data/emotion_history.csv"

def save_to_journal(emotion, confidences):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data = {"timestamp": timestamp, "emotion": emotion, **confidences}
    df = pd.DataFrame([data])
    os.makedirs("data", exist_ok=True)

    if not os.path.exists(JOURNAL_PATH) or os.stat(JOURNAL_PATH).st_size == 0:
        df.to_csv(JOURNAL_PATH, index=False)
    else:
        df.to_csv(JOURNAL_PATH, mode="a", header=False, index=False)

def show_journal():
    st.markdown("## 📖 Emotion History Journal")

    if os.path.exists(JOURNAL_PATH):
        try:
            df = pd.read_csv(JOURNAL_PATH)
            if df.empty:
                st.info("Your journal is currently empty.")
                return

            st.dataframe(df)
            st.line_chart(df.groupby("emotion").size())
        except EmptyDataError:
            st.info("Your journal file is present but contains no data.")
    else:
        st.info("No emotion history found yet.")