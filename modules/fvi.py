# ============================================================
# File: fvi.py
# Author: Mohammed Munazir
# Description: Implements feature variability index (FVI) analysis for understanding feedback variability
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px


def compute_fvi(df: pd.DataFrame, selected_features: list, group_by_col: str):
    fvi_records = []

    for feature in selected_features:
        try:
            sub_df = df[[group_by_col, feature]].dropna()

            grouped = sub_df.groupby(group_by_col)[feature].agg(['mean', 'count']).dropna()
            grouped.columns = ['Feature_Mean', 'Sample_Count']
            grouped["Weight"] = grouped["Sample_Count"]

            weighted_mean = np.average(grouped["Feature_Mean"], weights=grouped["Weight"])
            variance_numer = ((grouped["Feature_Mean"] - weighted_mean) ** 2 * grouped["Weight"]).sum()
            weighted_std = np.sqrt(variance_numer / grouped["Weight"].sum())

            if weighted_mean != 0:
                weighted_cv = weighted_std / abs(weighted_mean)
            else:
                weighted_cv = np.nan

            for group_val, row in grouped.iterrows():
                fvi_records.append({
                    "Feature": feature,
                    "Group": group_val,
                    "Group_By": group_by_col,
                    "Feature_Mean": round(row["Feature_Mean"], 4),
                    "Sample_Count": int(row["Sample_Count"]),
                    "FVI_Std": round(weighted_std, 4),
                    "FVI_CV": round(weighted_cv, 4),
                    "Weighted_Mean": round(weighted_mean, 4),
                    "Total_Samples": int(grouped["Weight"].sum())
                })

        except Exception as e:
            st.error(f"Error computing FVI for {feature} by {group_by_col}: {e}")

    return pd.DataFrame(fvi_records)



def plot_fvi_summary(fvi_df: pd.DataFrame):
    fig = px.bar(
        fvi_df.drop_duplicates(subset=["Feature", "Group_By"]),
        x="Feature",
        y="FVI_Std",
        color="FVI_CV",
        color_continuous_scale="Inferno",
        title="Feature Variability Index (FVI): Std & CV by Grouping"
    )
    fig.update_layout(
        xaxis_title="Feature",
        yaxis_title="FVI (Weighted Std Dev)",
        coloraxis_colorbar=dict(title="FVI (CV)")
    )
    st.plotly_chart(fig, use_container_width=True)



def plot_temporal_fvi_volatility(df: pd.DataFrame, selected_features: list, group_by_col: str):
    df["Date"] = pd.to_datetime(df["Date"], errors='coerce')
    df = df.dropna(subset=["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    for feature in selected_features:
        try:
            monthly_fvi = (
                df.groupby(["Month", group_by_col])[feature].mean()
                .unstack()
                .std(axis=1)
                .reset_index(name="FVI_Monthly_Std")
            )

            fig = px.line(
                monthly_fvi,
                x="Month",
                y="FVI_Monthly_Std",
                title=f"Temporal Volatility of FVI for '{feature}' by {group_by_col}"
            )
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="FVI (Monthly Std Dev Across Groups)"
            )
            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.warning(f"Could not generate temporal volatility for {feature}: {e}")



def identify_variability_outliers(df: pd.DataFrame, feature_col="Feature", value_col="Feature_Mean", group_col="Group"):
    emotion_cols = ["happiness", "sadness", "anger", "fear", "surprise"]
    sentiment_col = "Sentiment Score"

    st.subheader("Groups with High Sentiment Variance or Negative Emotion Averages")

    try:
        pivot_df = df.pivot(index=group_col, columns=feature_col, values=value_col).reset_index()
        sentiment_stats = df[df[feature_col] == sentiment_col].groupby(group_col)[value_col].agg(['mean', 'std']).reset_index()
        sentiment_stats.columns = [group_col, 'Sentiment_Mean', 'Sentiment_Std']

        combined = pd.merge(sentiment_stats, pivot_df[[group_col] + emotion_cols], on=group_col, how='left')
        st.dataframe(combined.style.background_gradient(cmap="OrRd"))

        high_variance = combined[combined["Sentiment_Std"] > combined["Sentiment_Std"].mean() + combined["Sentiment_Std"].std()]
        high_negatives = combined[combined["Sentiment_Mean"] < 0]

        if not high_variance.empty:
            st.markdown("### Groups with **High Sentiment Variance**")
            st.dataframe(high_variance)

        if not high_negatives.empty:
            st.markdown("### Groups with **Overall Negative Sentiment**")
            st.dataframe(high_negatives)

    except Exception as e:
        st.warning(f"Could not compute variability outliers: {e}")

    st.markdown("""
    - High **variance** implies inconsistent experience across groups.
    - High **negative sentiment** indicates structural concerns or service failures.
    """)


def generate_fvi_report(fvi_df: pd.DataFrame):
    st.header("Feature Variability Index (FVI) Summary")

    summary_df = (
        fvi_df.drop_duplicates(subset=["Feature", "Group_By"])
        .loc[:, ["Feature", "Group_By", "Weighted_Mean", "FVI_Std", "FVI_CV"]]
        .rename(columns={
            "Weighted_Mean": "Mean",
            "FVI_Std": "Std (Spatial)",
            "FVI_CV": "CV (Relative Variability)"
        })
    )

    st.dataframe(summary_df.style.format({
        "Mean": "{:.4f}",
        "Std (Spatial)": "{:.4f}",
        "CV (Relative Variability)": "{:.4f}"
    }))


def append_fvi_to_enriched(enriched_df: pd.DataFrame, fvi_df: pd.DataFrame) -> pd.DataFrame:
    try:
        merged_df = enriched_df.copy()
        
        for feature in fvi_df["Feature"].unique():
            temp_fvi = fvi_df[fvi_df["Feature"] == feature]
            group_by = temp_fvi["Group_By"].iloc[0]

         
            temp_fvi = temp_fvi[["Group", "FVI_Std", "FVI_CV"]].rename(
                columns={
                    "Group": group_by,
                    "FVI_Std": f"{feature}_FVI_Std",
                    "FVI_CV": f"{feature}_FVI_CV"
                }
            )

            
            merged_df = pd.merge(merged_df, temp_fvi, how="left", on=group_by)

        return merged_df
    except Exception as e:
        st.error(f"Failed to append FVI results to enriched data: {e}")
        return enriched_df
