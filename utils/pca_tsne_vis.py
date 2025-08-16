# ============================================================
# File: pca_tsne_vis.py
# Author: Mohammed Munazir
# Description: This module provides functions to visualize embeddings using PCA and t-SNE.
# ============================================================

import pandas as pd
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import streamlit as st

def visualize_embeddings(filtered_df: pd.DataFrame, embedder: SentenceTransformer , key="default"):
    st.subheader(" Visualize Aspect Embeddings")

    if len(filtered_df) < 3:
        st.info("Add more rows or adjust filters to enable embedding visualization.")
        return

    reduction_type = st.radio("Dimensionality Reduction Method", ["PCA", "t-SNE"])
    dim = st.slider("Output Dimensions", 2, 3, 2)

    with st.spinner("Computing embeddings..."):
        embeddings = embedder.encode(filtered_df["Aspect"].tolist())
        embeddings_scaled = StandardScaler().fit_transform(embeddings)

        if reduction_type == "PCA":
            reducer = PCA(n_components=dim)
        else:
            reducer = TSNE(
                n_components=dim,
                perplexity=5,
                init="random",
                learning_rate="auto",
                max_iter=1000
            )

        reduced = reducer.fit_transform(embeddings_scaled)
        reduced_df = pd.DataFrame(reduced, columns=[f"Dim{i+1}" for i in range(dim)])
        reduced_df["Sentiment"] = filtered_df["Sentiment"].values
        reduced_df["Aspect"] = filtered_df["Aspect"].values

        if dim == 2:
            fig = px.scatter(
                reduced_df,
                x="Dim1",
                y="Dim2",
                color="Sentiment",
                hover_data=["Aspect"],
                color_discrete_map={"POSITIVE": "green", "NEUTRAL": "gray", "NEGATIVE": "red"},
                title="Aspect Embeddings (2D)"
            )
        else:
            fig = px.scatter_3d(
                reduced_df,
                x="Dim1",
                y="Dim2",
                z="Dim3",
                color="Sentiment",
                hover_data=["Aspect"],
                color_discrete_map={"POSITIVE": "green", "NEUTRAL": "gray", "NEGATIVE": "red"},
                title="Aspect Embeddings (3D)"
            )

        st.plotly_chart(fig, use_container_width=True)
