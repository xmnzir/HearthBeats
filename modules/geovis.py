# ============================================================
# File: geovis.py
# Author: Mohammed Munazir
# Description: Implements Geo-Visualization
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

def run_geo_analysis(df: pd.DataFrame):
    st.header("Geo-Aware Analysis")

    if "Location" not in df.columns:
        st.warning(" 'Location' column is missing from your dataset.")
        return

    tab1, tab2, tab3= st.tabs([
        "Sentiment Map",
        "Emotion Heatmap",
        "Topic by Region",
        
    ])


    with tab1:
        st.subheader("Average Sentiment Score by Location")
        if "Sentiment Score" in df.columns:
            sent_map = df.groupby("Location")["Sentiment Score"].mean().reset_index()
            fig = px.bar(sent_map, x="Location", y="Sentiment Score", color="Sentiment Score", color_continuous_scale="RdYlGn")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("'Sentiment Score' not found in dataset.")


    with tab2:
        st.subheader("Emotion Distribution by Location")
        emotion_cols = ["happiness", "sadness", "anger", "fear", "surprise"]
        available_emotions = [col for col in emotion_cols if col in df.columns]
        if available_emotions:
            emo_df = df.groupby("Location")[available_emotions].mean().reset_index()
            emo_df_melt = emo_df.melt(id_vars="Location", var_name="Emotion", value_name="Score")
            fig = px.bar(emo_df_melt, x="Location", y="Score", color="Emotion", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No emotion columns found (happiness, sadness, etc.).")


    with tab3:
        st.subheader("Top Topics by Location")
        if "Detected Topics" in df.columns:
            topic_counts = df.groupby(["Location", "Detected Topics"]).size().reset_index(name="Count")
            fig = px.bar(topic_counts, x="Location", y="Count", color="Detected Topics", barmode="stack")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("'Detected Topics' column not found.")

  
