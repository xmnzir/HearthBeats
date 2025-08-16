# ============================================================
# File: invisible_voices.py
# Author: Mohammed Munazir
# Description: Implements and identifies underrepresented groups in the dataset.
# ============================================================

import pandas as pd
import streamlit as st
import plotly.express as px


def audit_data_overview(df: pd.DataFrame):
    st.header("Dataset Overview")

    st.subheader("Raw Data")
    st.dataframe(df, use_container_width=True)

    st.subheader("Column Summary")
    summary = pd.DataFrame({
        "Column": df.columns,
        "Type": [df[col].dtype for col in df.columns],
        "Missing (%)": [df[col].isnull().mean() * 100 for col in df.columns],
        "Unique Values": [df[col].nunique() for col in df.columns]
    })
    st.dataframe(summary)

    st.subheader("Class Imbalance Detection")
    cat_cols = [col for col in df.columns if df[col].dtype == "object" or df[col].nunique() <= 20]

    if not cat_cols:
        st.info("No categorical or low-cardinality columns found to evaluate imbalance.")
        return

    for col in cat_cols:
        st.markdown(f"#### `{col}` distribution")

        if df[col].dropna().empty:
            st.warning(f" Column `{col}` has no valid (non-null) data. Skipping.")
            continue

        value_counts = df[col].value_counts(normalize=True)
        if value_counts.empty:
            st.warning(f" Column `{col}` has no countable data. Skipping.")
            continue

        value_counts_df = value_counts.sort_values(ascending=False).reset_index()
        value_counts_df.columns = [col, "Proportion"]

        fig = px.bar(value_counts_df, x=col, y="Proportion", title=f"Distribution of {col}")
        st.plotly_chart(fig, use_container_width=True, key=f"{col}_dist_plot")

        if value_counts.iloc[0] > 0.75:
            st.warning(
                f"Potential imbalance in `{col}` — "
                f"'{value_counts.index[0]}' makes up {value_counts.iloc[0]*100:.1f}% of values."
            )


def invisible_voices_insights(df: pd.DataFrame, group_col="Location", sentiment_col="Sentiment",
                               topic_col="Detected Topics", optional_columns=None):
    if optional_columns is None:
        optional_columns = [
            "Type of Survey", "Customer ID", "Property Type", "Ethnicity",
            "Disability", "Question", "Tenure Type", "Age Group"
        ]

    st.header("Invisible Voices Spotlight")

    with st.spinner("Analyzing underrepresented groups..."):

        def analyze_underrepresented_groups(col):
            st.subheader(f"Underrepresentation in `{col}`")

            if df[col].dropna().empty:
                st.warning(f"Column `{col}` has no valid (non-null) data. Skipping analysis.")
                return

            group_counts = df[col].value_counts(normalize=True).reset_index()
            group_counts.columns = [col, "Proportion"]
            st.dataframe(group_counts.style.format({"Proportion": "{:.2%}"}))

            fig = px.bar(group_counts, x=col, y="Proportion", title=f"Group Representation in {col}")
            st.plotly_chart(fig, use_container_width=True, key=f"{col}_distribution")

            top_under = group_counts.sort_values("Proportion").head(3)

            for _, row in top_under.iterrows():
                subgroup = row[col]
                group_df = df[df[col] == subgroup]

                with st.expander(f" Spotlight: `{subgroup}` in `{col}` ({row['Proportion']:.2%})"):
                    st.write(f"🔹 **Total responses**: {len(group_df)}")

                    if sentiment_col in group_df.columns:
                        sentiment_dist = group_df[sentiment_col].value_counts(normalize=True).reset_index()
                        sentiment_dist.columns = [sentiment_col, "Proportion"]

                        tab1, tab2 = st.tabs(["Sentiment", " Topics"])
                        with tab1:
                            st.dataframe(sentiment_dist.style.format({"Proportion": "{:.2%}"}))
                        with tab2:
                            if topic_col in group_df.columns:
                                topic_freq = group_df[topic_col].value_counts().reset_index()
                                topic_freq.columns = [topic_col, "Count"]
                                st.dataframe(topic_freq.head(10))

                    if "Content" in group_df.columns and sentiment_col in group_df.columns:
                        negative_texts = group_df[group_df[sentiment_col].str.lower() == "negative"]
                        negative_texts = negative_texts["Content"].dropna()

                        if not negative_texts.empty:
                            sample_texts = negative_texts.sample(min(3, len(negative_texts)))
                            st.markdown("#### Sample Negative Feedback")
                            for i, text in enumerate(sample_texts):
                                st.markdown(f"**Example {i+1}:** {text}")

                            download_df = pd.DataFrame({"Negative Feedback": negative_texts.values})
                            st.download_button(
                                label="Download All Negative Feedback",
                                data=download_df.to_csv(index=False),
                                file_name=f"{col}_{subgroup}_negative_feedback.csv",
                                mime="text/csv"
                            )

                    st.markdown("#### Ethical Note")
                    st.markdown(f"""
                    - `{subgroup}` in `{col}` = **{row['Proportion']:.2%}** of all responses.
                    - Consider attention to this group in strategy, communication, and inclusion.
                    """)

        for col in optional_columns:
            if col in df.columns and df[col].nunique() > 1:
                analyze_underrepresented_groups(col)

    st.header("Geo-aware Representation Insights")
    if group_col in df.columns and df[group_col].nunique() > 1:
        analyze_underrepresented_groups(group_col)
