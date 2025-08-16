import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

from utils.data_loader import load_data, map_columns_ui
from modules.eda import run_eda
from modules.sentiment_analysis import analyze_sentiment
from modules.absa import run_absa_analysis
from modules.emotion import run_emotion_detection
from modules.topic_modeling import run_topic_modeling
from modules.invisible_voices import audit_data_overview, invisible_voices_insights
from modules.geovis import run_geo_analysis
from modules.geomap import run_geo_topic_module, run_nearby_infra_tab, run_weather_tab

from modules.fvi import (
    compute_fvi,
    plot_fvi_summary,
    plot_temporal_fvi_volatility,
    identify_variability_outliers,
    generate_fvi_report,
)


st.set_page_config(page_title="Housing Feedback NLP", layout="wide")


st.markdown("<hr style='margin-top: 0; margin-bottom: 20px;'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1.5, 6, 1.5])
with col1:
    st.markdown("<p style='text-align: left; font-size: 14px; color: gray;'>Research for</p>", unsafe_allow_html=True)
    st.image("assets/newcastle.png", width=140)

with col2:
    st.markdown("<h1 style='text-align: center; margin-bottom: 0;'>HearthBeats</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size: 18px; color: gray; margin-top: 0;'>"
        "The Housing Feedback Analysis Framework</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: left; font-size: 16px; color: #444;'>"
        "Gain insights from tenant feedback, uncover sentiment patterns, and help shape better housing experiences through natural language processing."
        "</p>",
        unsafe_allow_html=True,
    )

with col3:
    st.markdown("<p style='text-align: left; font-size: 14px; color: gray;'>In collaboration with</p>", unsafe_allow_html=True)
    st.image("assets/northstar.png", width=140)

st.markdown("<hr style='margin: 30px 0;'>", unsafe_allow_html=True)


with st.expander(" Upload & Map Your Data", expanded=True):
    uploaded_file = st.file_uploader("Upload your housing feedback CSV", type=["csv"])
    if uploaded_file:
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state["df"] = df

            column_mapping = map_columns_ui(df)
            if column_mapping:
                rename_map = {user_col: internal_name for internal_name, user_col in column_mapping.items()}
                df.rename(columns=rename_map, inplace=True)

                if "Survey Type" not in df.columns:
                    df["Survey Type"] = "prime"

                st.session_state["df_mapped"] = df
                st.success(" Data uploaded and columns mapped successfully.")
            else:
                st.warning(" Column mapping not complete.")
    else:
        st.info("⬆ Please upload a CSV file to begin.")


if "df_mapped" in st.session_state:
    df = st.session_state["df_mapped"]

    
    st.session_state.setdefault("eda_done", False)
    st.session_state.setdefault("bsa_df", None)
    st.session_state.setdefault("absa_df", None)
    st.session_state.setdefault("enriched_df", None)
    st.session_state.setdefault("topic_done", False)
    st.session_state.setdefault("sentiment_choice", None)
    st.session_state.setdefault("emotion_done", False)
    st.session_state.setdefault("geo_done", False)
    st.session_state.setdefault("ethical_done", False)
    st.session_state.setdefault("fvi_done", False)

    
    st.sidebar.title(" Choose a Module")

    with st.sidebar.expander("Step 1: Exploratory Data Analysis", expanded=False):
        if st.button("Run EDA"):
            st.session_state["eda_done"] = True

    with st.sidebar.expander("Step 2: Sentiment Analysis"):
        if st.button("Run Sentiment Analysis"):
            st.session_state["sentiment_choice"] = "Sentiment Analysis"
        if st.button("Run Aspect-Based Sentiment Analysis (ABSA)"):
            st.session_state["sentiment_choice"] = "ABSA"

    with st.sidebar.expander("Step 3: Emotion Detection"):
        if st.button("Run Emotion Detection"):
            st.session_state["emotion_done"] = True

    with st.sidebar.expander("Step 4: Topic Modeling"):
        if st.button("Run Topic Modeling"):
            st.session_state["topic_done"] = True


    with st.sidebar.expander("Step 5: Feature Variability Index"):
        if st.button("Run FVI"):
            st.session_state["fvi_done"] = True

    
    with st.sidebar.expander("Step 6: Geo-Aware Analysis"):
         if st.button("Run Geo Module"):
            st.session_state["geo_done"] = True
            

    with st.sidebar.expander("Step 7: Ethical & Fairness Audit"):
        if st.button("Run Ethical Audit"):
            st.session_state["ethical_done"] = True

   

    if st.session_state["eda_done"]:
        with st.expander("Exploratory Data Analysis Results", expanded=False):
            with st.spinner("Running Exploratory Data Analysis..."):
                run_eda(df)

    if st.session_state["sentiment_choice"]:
        with st.expander("Sentiment Analysis Results", expanded=False):
            with st.spinner("Analyzing Sentiment..."):
                if st.session_state["sentiment_choice"] == "Sentiment Analysis":
                    bsa_df = analyze_sentiment(df)
                    st.session_state["bsa_df"] = bsa_df
                    st.dataframe(bsa_df)
                elif st.session_state["sentiment_choice"] == "ABSA":
                    absa_df = run_absa_analysis(df)
                    st.session_state["absa_df"] = absa_df
                    st.dataframe(absa_df)

    if st.session_state["emotion_done"]:
        with st.expander("Emotion Detection Results", expanded=False):
            absa_df = st.session_state.get("absa_df")
            bsa_df = st.session_state.get("bsa_df")

            input_df = absa_df if absa_df is not None and not absa_df.empty else bsa_df
            if input_df is not None:
                with st.spinner("Detecting Emotions..."):
                    emotion_df = run_emotion_detection(input_df)
                    if emotion_df is not None and not emotion_df.empty:
                        emotion_scores = emotion_df.drop(columns=["text"], errors="ignore")
                        input_df = input_df.reset_index(drop=True)
                        enriched_df = pd.concat([input_df, emotion_scores], axis=1)
                        st.session_state["enriched_df"] = enriched_df
                        st.dataframe(enriched_df)
                        st.download_button(
                            "⬇Download Results with Emotions",
                            enriched_df.to_csv(index=False),
                            "enriched_emotions.csv",
                            "text/csv"
                        )
                    else:
                        st.warning("No emotion scores generated.")
            else:
                st.warning("Run Sentiment or ABSA analysis first.")

    if st.session_state["topic_done"]:
        with st.expander("Topic Modeling Results", expanded=False):
            enriched_df = st.session_state.get("enriched_df")
            if enriched_df is not None and not enriched_df.empty:
                if "Content" not in enriched_df.columns:
                    st.error(" 'Content' column not found.")
                else:
                    with st.spinner("Running Topic Modeling..."):
                        topics_names, topic_summary, topic_model = run_topic_modeling(enriched_df, text_col="Content")

                        enriched_df["Detected Topics"] = topics_names
                        st.session_state["enriched_df"] = enriched_df

                        st.subheader(" Topic Frequency")
                        fig, ax = plt.subplots(figsize=(10, 5))
                        sns.barplot(x="Frequency", y="Topic", data=topic_summary, ax=ax, palette="coolwarm")
                        st.pyplot(fig)

                        st.subheader(" BERTopic Interactive Visualizations")
                        components.html(topic_model.visualize_barchart().to_html(), height=600)
                        components.html(topic_model.visualize_topics().to_html(), height=600)
                        components.html(topic_model.visualize_term_rank().to_html(), height=600)
                        components.html(topic_model.visualize_hierarchy().to_html(), height=600)
                        components.html(topic_model.visualize_heatmap().to_html(), height=600)

                        st.download_button(
                            "Download Full Results with Topics",
                            enriched_df.to_csv(index=False),
                            "full_enriched_results.csv",
                            "text/csv"
                        )
            else:
                st.warning("Run Emotion Detection first.")

    if st.session_state["fvi_done"]:
        with st.expander(" Feature Variability Index", expanded=True):
            enriched_df = st.session_state.get("enriched_df")
            if enriched_df is not None and not enriched_df.empty:
                numeric_cols = enriched_df.select_dtypes(include='number').columns.tolist()
                groupable_cols = [col for col in enriched_df.columns if enriched_df[col].nunique() < 50 and col != "Date"]

                selected_features = st.multiselect("Select features for FVI", numeric_cols)
                if "Date" in enriched_df.columns:
                    timewise = st.checkbox("Include time-based volatility", value=True)
                else:
                    timewise = st.checkbox("Include time-based volatility", value=False, disabled=True)

                group_by_col = st.selectbox("Select grouping dimension", options=groupable_cols, index=groupable_cols.index("Location") if "Location" in groupable_cols else 0)

                if selected_features:
                    with st.spinner("Computing Feature Volatility Index..."):
                        fvi_df = compute_fvi(enriched_df, selected_features, group_by_col)
                        st.dataframe(fvi_df)
                        plot_fvi_summary(fvi_df)
                        identify_variability_outliers(fvi_df)
                        if timewise:
                            plot_temporal_fvi_volatility(enriched_df, selected_features, group_by_col)
                        generate_fvi_report(fvi_df)

                        st.markdown(" **Interpretation:** High volatility means uneven distribution across groups. CV helps compare features on different scales.")

                        st.download_button(
                            label="⬇ Download FVI Results",
                            data=fvi_df.to_csv(index=False),
                            file_name="feature_volatility_index.csv",
                            mime="text/csv"

                        
                        )
                else:
                    st.info("Select at least one numeric feature.")
            else:
                st.warning("Run earlier modules to enrich the data first.")

    if st.session_state["geo_done"]:
        with st.expander(" Geo-Aware Insights", expanded=True):
            enriched_df = st.session_state.get("enriched_df")
            if enriched_df is not None and not enriched_df.empty:
                with st.spinner("Running Geo-Aware Analysis..."):
                    tabs = st.tabs(["Geo Analysis", "Geo Topic Module", "Nearby Infrastructure", "Weather Metrics"])

                    with tabs[0]:
                        run_geo_analysis(enriched_df)

                    with tabs[1]:
                        run_geo_topic_module(enriched_df)

                    with tabs[2]:
                        run_nearby_infra_tab(enriched_df)

                    with tabs[3]:
                        run_weather_tab(enriched_df)

    if st.session_state["ethical_done"]:
        with st.expander("Ethical & Fairness Audit Results", expanded=True):
            enriched_df = st.session_state.get("enriched_df")
            if enriched_df is not None and not enriched_df.empty and st.session_state["topic_done"]:
                with st.spinner("Running Ethical Audit..."):
                    audit_data_overview(enriched_df)
                    invisible_voices_insights(
                        enriched_df,
                        group_col="Location",
                        sentiment_col="Sentiment",
                        topic_col="Detected Topics",
                        optional_columns=["Type of Survey", "Customer ID", "Property Type", "Ethnicity", "Disability", "Question", "Tenure Type", "Age Group"]
                    )
                    st.success("Dataset overview and class imbalance check completed.")
            else:
                st.warning("Run all prior steps before ethical audit.")
