# ============================================================
# File: emotion.py
# Author: Mohammed Munazir
# Description: Implements Emotion Detection Module
# ============================================================

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import pandas as pd
import streamlit as st
import plotly.express as px


model_to_ekman = {
    "joy": "happiness",
    "love": "happiness",
    "sadness": "sadness",
    "anger": "anger",
    "fear": "fear",
    "surprise": "surprise",
}

ekman_emotions = ["happiness", "sadness", "anger", "fear", "surprise"]

class EmotionDetector:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.model = AutoModelForSequenceClassification.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
        self.model.to(self.device)
        self.model.eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    def predict_emotions(self, texts):
        results = []
        progress_bar = st.progress(0)
        total = len(texts)

        for i, text in enumerate(texts):
            if not isinstance(text, str) or len(text.strip()) == 0:
                probs_dict = {emotion: 0.0 for emotion in ekman_emotions}
            else:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

                probs_dict = {emotion: 0.0 for emotion in ekman_emotions}
                for idx, label in enumerate(self.labels):
                    ekman_label = model_to_ekman.get(label.lower(), None)
                    if ekman_label:
                        probs_dict[ekman_label] += probs[idx]

            probs_dict["text"] = text
            results.append(probs_dict)
            progress_bar.progress((i + 1) / total)

        progress_bar.empty()
        return pd.DataFrame(results)


def run_emotion_detection(df):
    st.markdown("## Emotion Detection")

    if "Content" not in df.columns:
        st.error("Column 'Content' not found in dataframe for emotion detection.")
        return

    st.markdown("### Select Row Range for Emotion Analysis")
    row_range = st.slider(
        "Select range of rows to analyze",
        min_value=0,
        max_value=len(df),
        value=(0, min(100, len(df))),
        step=1
    )

    lower, upper = row_range
    sample_df = df["Content"].astype(str).iloc[lower:upper]

    detector = EmotionDetector()
    emotion_df = detector.predict_emotions(sample_df.tolist())


    emotion_columns = [col for col in emotion_df.columns if col != "text"]
    emotion_df["Top Emotion"] = emotion_df[emotion_columns].idxmax(axis=1)

    st.markdown("### Emotion Scores")
    st.dataframe(emotion_df)

    
    combined_df = df.copy()
    for col in emotion_df.columns:
        if col != "text":
            combined_df.loc[sample_df.index, col] = emotion_df[col].values

    st.markdown("### Average Emotion Scores")
    mean_scores = emotion_df[emotion_columns].mean().sort_values(ascending=False)
    st.bar_chart(mean_scores)

 
    st.subheader("Emotion Distribution (Top Emotion)")
    dist_counts = emotion_df["Top Emotion"].value_counts()
    pie_fig = px.pie(
        names=dist_counts.index,
        values=dist_counts.values,
        title="Distribution of Top Emotions",
        color=dist_counts.index,
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    st.plotly_chart(pie_fig, use_container_width=True)

 
    if "Location" in df.columns:
        st.subheader("Average Emotion by Location")
        loc_avg = combined_df.dropna(subset=["Location"])[["Location"] + emotion_columns].groupby("Location").mean().reset_index()
        for emotion in emotion_columns:
            fig = px.bar(
                loc_avg, x="Location", y=emotion,
                title=f"Avg {emotion.capitalize()} by Location",
                color=emotion, color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, use_container_width=True)


    if "Age Group" in df.columns:
        st.subheader("Average Emotion by Age Group")
        age_avg = combined_df.dropna(subset=["Age Group"])[["Age Group"] + emotion_columns].groupby("Age Group").mean().reset_index()
        for emotion in emotion_columns:
            fig = px.bar(
                age_avg, x="Age Group", y=emotion,
                title=f"Avg {emotion.capitalize()} by Age Group",
                color=emotion, color_continuous_scale="Plasma"
            )
            st.plotly_chart(fig, use_container_width=True)


    st.session_state["emotion_df"] = emotion_df
    st.session_state["emotion_merged_df"] = combined_df


    csv_data = combined_df.to_csv(index=False)
    st.download_button(
        label="⬇Download Dataset with Emotion Scores",
        data=csv_data,
        file_name="emotion_results.csv",
        mime="text/csv"
    )

    return emotion_df
