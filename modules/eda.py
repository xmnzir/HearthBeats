# ============================================================
# File: eda.py
# Author: Mohammed Munazir
# Description: Implement Exploratory Data Analysis (EDA)
# ============================================================

import streamlit as st
import pandas as pd
import plotly.express as px

def run_eda(df: pd.DataFrame):
    st.header("Exploratory Data Analysis")

    with st.expander("Preview Data"):
        st.write(df.head())

    with st.expander("Missing Value Summary"):
        st.write(df.isnull().sum())


    if "Location" in df:
        st.subheader("Location Distribution")
        fig = px.histogram(df, x="Location", color="Location")
        st.plotly_chart(fig, use_container_width=True)

    if "Date" in df:
        st.subheader("Responses Over Time")
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        date_counts = df["Date"].value_counts().sort_index()
        st.line_chart(date_counts)

    if "Age Group" in df:
        st.subheader("Age Group Distribution")
        fig = px.histogram(df, x="Age Group", color="Age Group")
        st.plotly_chart(fig, use_container_width=True)

    if "Ethnicity" in df:
        st.subheader("Ethnicity Breakdown")
        fig = px.histogram(df, x="Ethnicity", color="Ethnicity")
        st.plotly_chart(fig, use_container_width=True)

    if "Property Type" in df:
        st.subheader("Property Type Distribution")
        fig = px.histogram(df, x="Property Type", color="Property Type")
        st.plotly_chart(fig, use_container_width=True)

    if "Disability" in df:
        st.subheader("Disability Disclosure")
        fig = px.histogram(df, x="Disability", color="Disability")
        st.plotly_chart(fig, use_container_width=True)

    if "Question" in df:
        st.subheader("Question Category Distribution")
        fig = px.histogram(df, x="Question", color="Question")
        st.plotly_chart(fig, use_container_width=True)

    if "Content" in df:
        st.subheader("Text Length Distribution")
        df["Content Length"] = df["Content"].astype(str).apply(len)
        fig = px.histogram(df, x="Content Length", nbins=50)
        st.plotly_chart(fig, use_container_width=True)


    if "Survey Type" in df and "Location" in df:
        st.subheader("Survey Type by Location")
        fig = px.histogram(df, x="Location", color="Survey Type", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    if "Survey Type" in df and "Property Type" in df:
        st.subheader("Survey Type by Property Type")
        fig = px.histogram(df, x="Property Type", color="Survey Type", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    if "Date" in df and "Survey Type" in df:
        st.subheader("Survey Counts Over Time by Type")
        df_grouped = df.groupby(["Date", "Survey Type"]).size().reset_index(name="Counts")
        fig = px.line(df_grouped, x="Date", y="Counts", color="Survey Type")
        st.plotly_chart(fig, use_container_width=True)

    if "Tenure Type" in df:
        st.subheader("Tenure Type Distribution")
        fig = px.histogram(df, x="Tenure Type", color="Tenure Type")
        st.plotly_chart(fig, use_container_width=True)

    if "Age Group" in df and "Survey Type" in df:
        st.subheader("Age Group by Survey Type")
        fig = px.histogram(df, x="Age Group", color="Survey Type", barmode="group")
        st.plotly_chart(fig, use_container_width=True)

    if "Survey Type" in df:
        st.subheader("Survey Type Distribution")
        fig = px.histogram(df, x="Survey Type", color="Survey Type")
        st.plotly_chart(fig, use_container_width=True)
