# ============================================================
# File: absa.py
# Author: Mohammed Munazir
# Description: Implements Aspect-Based Sentiment Analysis (ABSA)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import spacy
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from utils.text_preprocessing import extract_aspects_spacy, clean_text
from utils.pca_tsne_vis import visualize_embeddings  
from transformers import AutoTokenizer

@st.cache_resource(show_spinner=False)
def load_absa_pipeline():
    model_name = "yangheng/deberta-v3-base-absa-v1.1"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)  
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        return classifier
    except Exception as e:
        st.error(f"Error loading ABSA model: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except Exception as e:
        st.error(f"Error loading spaCy: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

def run_absa_analysis(df: pd.DataFrame):
    st.header("Aspect-Based Sentiment Analysis")

    if "Answer" in df.columns:
        original_text_col = "Answer"
        df = df.rename(columns={original_text_col: "Content"})
    elif "Content" in df.columns:
        original_text_col = "Content"
    else:
        st.warning("No 'Answer' or 'Content' column found.")
        return

    has_location = "Location" in df.columns

    nlp = load_spacy_model()
    classifier = load_absa_pipeline()
    embedder = load_embedding_model()

    if classifier is None or nlp is None or embedder is None:
        st.warning("ABSA or embedding model failed to load.")
        return

    max_rows = st.slider("Select number of rows to analyze", 10, len(df), value=min(100, len(df)))
    df_sample = df.head(max_rows).copy()
    content_series = df_sample["Content"].dropna().astype(str)
    content_series = content_series.apply(clean_text)

    rows = []
    progress = st.progress(0)

    with st.spinner("Extracting aspects and sentiments..."):
        for i, (idx, text) in enumerate(content_series.items()):
            location = df_sample.loc[idx, "Location"] if has_location else None
            aspects = extract_aspects_spacy(text, nlp)
            for aspect in aspects:
                try:
                    result = classifier(text, text_pair=aspect)[0]
                    rows.append({
                        "Index": idx,
                        "Content": text,
                        "Aspect": aspect,
                        "Sentiment": result["label"].upper(),
                        "Confidence": round(result["score"], 3),
                        "Location": location
                    })
                except Exception as e:
                    st.error(f"Error at index {idx}: {e}")
            progress.progress((i + 1) / len(content_series))

    absa_df = pd.DataFrame(rows)
    if absa_df.empty:
        st.warning("No aspects found.")
        return

    def sentiment_score(row):
        return row["Confidence"] if row["Sentiment"] == "POSITIVE" else -row["Confidence"] if row["Sentiment"] == "NEGATIVE" else 0

    absa_df["Sentiment Score"] = absa_df.apply(sentiment_score, axis=1)

    aggregated = absa_df.groupby("Index").agg({
        "Sentiment Score": "mean",
        "Confidence": "mean",
        "Aspect": lambda x: ", ".join(x.unique()),
    }).rename(columns={
        "Sentiment Score": "Sentiment Score",
        "Confidence": "Avg_Confidence"
    }).reset_index()

    
    def label_sentiment(score):
        if score > 0:
            return "POSITIVE"
        elif score < 0:
            return "NEGATIVE"
        else:
            return "NEUTRAL"

    aggregated["Sentiment"] = aggregated["Sentiment Score"].apply(label_sentiment)

    df_sample_reset = df_sample.reset_index()
    merged_df = df_sample_reset.merge(aggregated, left_index=True, right_on="Index", how="left")

    merged_df["Sentiment Score"].fillna(0, inplace=True)
    merged_df["Avg_Confidence"].fillna(0, inplace=True)
    merged_df["Aspect"].fillna("", inplace=True)
    merged_df["Sentiment"].fillna("NEUTRAL", inplace=True)

    if original_text_col != "Content":
        merged_df.rename(columns={"Content": original_text_col}, inplace=True)

    st.subheader("ABSA Results (Aspect-level)")
    st.dataframe(absa_df, use_container_width=True)

    st.subheader("ABSA Results (Aggregated per row)")
    st.dataframe(merged_df, use_container_width=True)

    st.download_button(
        label="Download Processed ABSA Results",
        data=merged_df.to_csv(index=False),
        file_name="absa_aggregated_results.csv",
        mime="text/csv"
    )

    st.subheader("Filter by Aspect")
    if "reset" not in st.session_state:
        st.session_state.reset = False

    if st.button("Reset Filters"):
        st.session_state.reset = True

    if st.session_state.reset:
        aspect_query = ""
        tag_selection = []
        st.session_state.reset = False
    else:
        aspect_query = st.text_input("Search for Aspect (partial match)")
        top_n = st.slider("Show Top N Most Frequent Aspects", 5, 50, 20)
        top_aspects = absa_df["Aspect"].value_counts().head(top_n).index.tolist()
        tag_selection = st.multiselect("Popular Aspects (checkbox filter)", options=top_aspects)

    if aspect_query:
        filtered_df = absa_df[absa_df["Aspect"].str.contains(aspect_query, case=False, na=False)]
    elif tag_selection:
        filtered_df = absa_df[absa_df["Aspect"].isin(tag_selection)]
    else:
        filtered_df = absa_df.copy()

    st.subheader("Aspect Frequency (Table)")
    aspect_freq_df = filtered_df["Aspect"].value_counts().reset_index()
    aspect_freq_df.columns = ["Aspect", "Frequency"]
    st.dataframe(aspect_freq_df, use_container_width=True)
    
    st.subheader("Sentiment Distribution (Table)")
    sentiment_dist_df = filtered_df.groupby(["Aspect", "Sentiment"]).size().reset_index(name="Count")
    sentiment_pivot = sentiment_dist_df.pivot(index="Aspect", columns="Sentiment", values="Count").fillna(0)
    st.dataframe(sentiment_pivot, use_container_width=True)
    st.subheader("Sentiment Distribution")



    st.subheader("Average Confidence by Sentiment")
    avg_conf = filtered_df.groupby("Sentiment")["Confidence"].mean().reset_index()
    conf_fig = px.bar(
        avg_conf, x="Sentiment", y="Confidence",
        color="Sentiment",
        color_discrete_map={"POSITIVE": "green", "NEUTRAL": "gray", "NEGATIVE": "red"}
    )
    st.plotly_chart(conf_fig, use_container_width=True)

    
    visualize_embeddings(filtered_df, embedder)

    return merged_df
