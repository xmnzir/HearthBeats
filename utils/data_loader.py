# ============================================================
# File: data_loader.py
# Author: Mohammed Munazir
# Description: Loads the data from a CSV file and maps columns
# ============================================================


import pandas as pd
import streamlit as st

def load_data(uploaded_file):
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f" Error loading file: {e}")
        return None

def map_columns_ui(df):
    st.markdown("###  Column Mapping")

    column_aliases = {
        "Survey": "Content",             
        "Location": "Location",          
        "Date": "Date",
        "Type of Survey": "Survey Type",  
        "Customer ID": "Customer ID",
        "Property Type": "Property Type",
        "Ethnicity": "Ethnicity",
        "Disability": "Disability",
        "Question": "Question",
        "Tenure Type": "Tenure Type",
        "Age Group": "Age Group"
    }

    required_columns = ["Survey", "Location", "Date"]
    optional_columns = ["Type of Survey", "Customer ID", "Property Type", "Ethnicity", "Disability", "Question", "Tenure Type", "Age Group"]

    user_mapping = {}

    for label, columns, required in [(" Required Columns", required_columns, True), (" Optional Columns", optional_columns, False)]:
        st.subheader(label)
        for col in columns:
            selected = st.selectbox(f"Select column for **{col}**", [None] + list(df.columns))
            if selected:
                user_mapping[column_aliases[col]] = selected
            elif required:
                st.error(f" Required column '{col}' not mapped.")
                return None

    st.success(" Column mapping complete.")
    return user_mapping

def apply_column_mapping(df: pd.DataFrame, user_mapping: dict) -> pd.DataFrame:
    
    rename_map = {v: k for k, v in user_mapping.items()}
    df = df.rename(columns=rename_map)

    if "Survey Type" not in df.columns:
        df["Survey Type"] = "prime"

    return df
