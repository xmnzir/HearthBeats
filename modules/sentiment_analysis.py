# ============================================================
# File: sentiment_analysis.py
# Author: Mohammed Munazir
# Description: Implements Sentiment analysis
# ============================================================

import streamlit as st
from transformers import pipeline
import pandas as pd
import plotly.express as px

@st.cache_resource(show_spinner=False)
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(df: pd.DataFrame):
    st.header("Sentiment Analysis")

    if "Answer" in df.columns:
        df = df.rename(columns={"Answer": "Content"})

    if "Content" not in df.columns:
        st.warning(" 'Answer' column not found in data. Please check your column mapping.")
        return df

    model = load_sentiment_model()

    max_rows = st.slider(
        "Select number of rows to analyze",
        min_value=10,
        max_value=len(df),
        value=min(100, len(df))
    )

    progress_bar = st.progress(0)
    batch_size = 20
    results = []

   
    sample_df = df["Content"].astype(str).head(max_rows)
    total_batches = (len(sample_df) + batch_size - 1) // batch_size

    with st.spinner("..."):
        for i in range(total_batches):
            start = i * batch_size
            end = min(start + batch_size, len(sample_df))
            batch_texts = sample_df.iloc[start:end].to_list()
            batch_results = model(batch_texts)
            results.extend(batch_results)
            progress_bar.progress((i + 1) / total_batches)

   
    df.loc[sample_df.index, "Sentiment"] = [r["label"] for r in results]
    df.loc[sample_df.index, "Sentiment Score"] = [
        r["score"] if r["label"] == "POSITIVE"
        else -r["score"] if r["label"] == "NEGATIVE"
        else 0
        for r in results
    ]

    st.success("Sentiment analysis complete.")

    
    sample_with_sentiment = df.loc[sample_df.index].copy()


    unique_sentiments = sample_with_sentiment["Sentiment"].dropna().astype(str).unique().tolist()
    selected_sentiments = st.multiselect(
        "Filter by Sentiment",
        options=unique_sentiments,
        default=unique_sentiments
    )
    filtered_df = sample_with_sentiment[sample_with_sentiment["Sentiment"].isin(selected_sentiments)]

   
   
    sentiment_emojis = {"POSITIVE": "", "NEGATIVE": "", "NEUTRAL": ""}
    sentiment_colors = {"POSITIVE": "green", "NEGATIVE": "red", "NEUTRAL": "gray"}

    counts = filtered_df["Sentiment"].value_counts().reindex(unique_sentiments).fillna(0)
    labels = [f"{sentiment_emojis.get(s, '')} {s}" for s in counts.index]

    st.subheader("Sentiment Distribution")
    pie_fig = {
        "data": [
            {
                "values": counts.values,
                "labels": labels,
                "type": "pie",
                "marker": {"colors": [sentiment_colors.get(s, "blue") for s in counts.index]},
                "textinfo": "label+percent"
            }
        ],
        "layout": {"margin": {"t": 0, "b": 0, "l": 0, "r": 0}}
    }
    st.plotly_chart(pie_fig, use_container_width=True)


    num_left_out = sample_with_sentiment["Sentiment"].isna().sum()
    if num_left_out > 0:
        st.markdown(f"**Rows left out from analyzing (empty/missing content):** {num_left_out}")

    
    avg_scores = filtered_df.groupby("Sentiment")["Sentiment Score"].mean().reindex(unique_sentiments).fillna(0)
    score_df = pd.DataFrame({
        "Sentiment": avg_scores.index,
        "Average Score": avg_scores.values,
        "Emoji": [sentiment_emojis.get(s, "") for s in avg_scores.index]
    })

    st.subheader("Average Sentiment Confidence Scores")
    bar_fig = px.bar(
        score_df,
        x="Sentiment",
        y="Average Score",
        text="Emoji",
        color="Sentiment",
        color_discrete_map=sentiment_colors,
        labels={"Average Score": "Avg Score (-1 to 1)"},
        range_y=[-1, 1]
    )
    bar_fig.update_traces(textposition="outside")
    st.plotly_chart(bar_fig, use_container_width=True)

    
    if "Location" in sample_with_sentiment.columns:
        st.subheader("Average Sentiment by Management Area")
        loc_score = sample_with_sentiment.dropna(subset=["Sentiment Score", "Location"]).groupby("Location")["Sentiment Score"].mean().reset_index()
        loc_fig = px.bar(
            loc_score,
            x="Location",
            y="Sentiment Score",
            color="Sentiment Score",
            color_continuous_scale="RdYlGn",
            labels={"Sentiment Score": "Avg Sentiment (-1 to 1)"},
            height=400
        )
        st.plotly_chart(loc_fig, use_container_width=True)

  
    if "Age Group" in sample_with_sentiment.columns:
        st.subheader("Average Sentiment by Age Group")
        age_score = sample_with_sentiment.dropna(subset=["Sentiment Score", "Age Group"]).groupby("Age Group")["Sentiment Score"].mean().reset_index()
        age_fig = px.bar(
            age_score,
            x="Age Group",
            y="Sentiment Score",
            color="Sentiment Score",
            color_continuous_scale="RdYlGn",
            labels={"Sentiment Score": "Avg Sentiment (-1 to 1)"},
            height=400
        )
        st.plotly_chart(age_fig, use_container_width=True)

    csv_data = filtered_df.to_csv(index=False)
    st.download_button("Download Filtered CSV with Sentiment", csv_data, "sentiment_results.csv", "text/csv")


    bsa_df = filtered_df
    return bsa_df
