
# ============================================================
# File: topic_modeling.py
# Author: Mohammed Munazir
# Description: Implements Bert topic modeling for identifying main topics and key themes
# ============================================================

import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import streamlit as st

from utils.pca_tsne_vis import visualize_embeddings


STOPWORDS_SET = {
    
    "the", "a", "an", "and", "or", "but", "if", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through", "during", "before", "after", "above",
    "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "can", "will", "just", "should", "now",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "myself",
    "yourself", "himself", "herself", "itself", "ourselves", "yourselves", "themselves",
    "", "none", "nan", "unknown", "misc", "miscellaneous",
    
    "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn",
    
    "also", "although", "always", "among", "anyway", "because", "become", "becomes", "becoming",
    "besides", "beyond", "certain", "concerning", "considering", "despite", "else",
    "elsewhere", "enough", "especially", "etc", "even", "ever", "every", "everyone", "everything",
    "far", "furthermore", "however", "indeed", "instead", "less", "like", "likely", "many",
    "moreover", "mostly", "much", "neither", "never", "noone", "nor", "often",
    "otherwise", "quite", "rather", "really", "relatively", "several", "somehow",
    "someone", "something", "sometimes", "somewhat", "still", "though", "thus",
    "together", "toward", "towards", "unless", "usually", "via", "well", "yet",
    
    "anybody", "anyone", "anything", "each", "either", "few", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how's", "i'd", "i'll",
    "i'm", "i've", "it'd", "it'll", "it's", "let's", "my", "myself", "nobody", "nothing", "one",
    "ones", "our", "ours", "ourselves", "someone", "something", "that", "that'll", "that's",
    "their", "theirs", "them", "themselves", "they'd", "they'll", "they're", "they've", "this",
    "those", "we'd", "we'll", "we're", "we've", "what", "what's", "whatever", "when", "when's",
    "where", "where's", "which", "who", "who's", "whom", "why", "why's", "would", "you'd",
    "you'll", "you're", "you've", "your", "yours"
}


def clean_text(text):
    """
    Basic text cleaning without spaCy:
    - Lowercase
    - Remove non-alphanumeric characters
    - Split into words
    - Remove stopwords
    """
    text = re.sub(r"\W+", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split() if word not in STOPWORDS_SET and len(word) > 2]
    return " ".join(tokens)


def clean_topic_name(name):
    if isinstance(name, (list, tuple)):
        name = ", ".join(name)
    return re.sub(r'^\d+_', '', str(name)).strip()


def is_weak_topic_name(name):
    tokens = re.split(r'[_ ,]+', name.lower())
    tokens = [t for t in tokens if t]
    stopword_count = sum(1 for t in tokens if t in STOPWORDS_SET)
    return not tokens or (stopword_count / len(tokens) >= 0.9)


def clean_and_filter_topic(name):
    name = clean_topic_name(name)
    tokens = re.split(r'[_ ,]+', name.lower())
    tokens = [t for t in tokens if t not in STOPWORDS_SET and len(t) > 2]
    return ", ".join(tokens) if tokens else None


def summarize_cluster_keywords(texts, top_n=5):
    vectorizer = CountVectorizer(stop_words='english', max_features=2000)
    X = vectorizer.fit_transform(texts)
    token_counts = X.sum(axis=0).A1
    tokens = vectorizer.get_feature_names_out()
    top_indices = token_counts.argsort()[::-1][:top_n]
    top_keywords = [tokens[i] for i in top_indices]
    return ", ".join(top_keywords)


def run_topic_modeling(df, text_col="Content", visualize=True):
    df = df.copy()
    raw_texts = df[text_col].fillna("").astype(str).tolist()
    texts = [clean_text(t) for t in raw_texts]

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    with st.spinner("Analyzing underrepresented groups..."):
        vectorizer_model = CountVectorizer(
            stop_words=list(STOPWORDS_SET),
            ngram_range=(1, 2),
            min_df=3,
            max_df=0.8,
            max_features=3000
        )

        topic_model = BERTopic(
            embedding_model=embedder,
            vectorizer_model=vectorizer_model,
            language="english",
            calculate_probabilities=False,
            verbose=True,
            min_topic_size=10,
        )

        topics, _ = topic_model.fit_transform(texts)

        topic_info = topic_model.get_topic_info()
        topic_info = topic_info[topic_info.Topic != -1].copy()
        topic_info["Name"] = topic_info["Name"].apply(clean_topic_name)
        topic_info = topic_info[~topic_info["Name"].apply(is_weak_topic_name)]

        topic_labels = {
            tid: clean_and_filter_topic(name)
            for tid, name in zip(topic_info["Topic"], topic_info["Name"])
            if clean_and_filter_topic(name)
        }

        final_labels = []
        unlabeled_indices = []

        for i, tid in enumerate(topics):
            label = topic_labels.get(tid)
            if label:
                final_labels.append(label)
            else:
                final_labels.append(None)
                unlabeled_indices.append(i)

        if unlabeled_indices:
            unlabeled_texts = [texts[i] for i in unlabeled_indices]
            embeddings = embedder.encode(unlabeled_texts)
            n_clusters = min(max(2, int(np.sqrt(len(unlabeled_indices)))), 10)

            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(embeddings)

            cluster_to_texts = {}
            for cl, txt in zip(kmeans_labels, unlabeled_texts):
                cluster_to_texts.setdefault(cl, []).append(txt)

            cluster_summaries = {
                cl: summarize_cluster_keywords(txts, top_n=5)
                for cl, txts in cluster_to_texts.items()
            }

            for idx, cl in zip(unlabeled_indices, kmeans_labels):
                fallback_label = clean_and_filter_topic(cluster_summaries[cl])
                final_labels[idx] = fallback_label if fallback_label else "Miscellaneous"

        df["Detected Topics"] = final_labels

        topic_summary = pd.DataFrame(final_labels, columns=["Topic"])
        topic_summary = topic_summary["Topic"].value_counts().reset_index()
        topic_summary.columns = ["Topic", "Frequency"]

    return final_labels, topic_summary, topic_model


# Visualization functions
def visualize_topics(topic_model):
    fig = topic_model.visualize_topics()
    fig.show()


def visualize_topic_distribution(topic_model, topics):
    fig = topic_model.visualize_distribution(topics)
    fig.show()


def visualize_hierarchy(topic_model):
    fig = topic_model.visualize_hierarchy()
    fig.show()


def visualize_heatmap(topic_model):
    fig = topic_model.visualize_heatmap()
    fig.show()


def explore_topic(topic_model, topic_id):
    fig = topic_model.visualize_barchart(topic_id)
    fig.show()
