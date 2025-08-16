# ============================================================
# File: fvi_reliability_check.py
# Author: Mohammed Munazir
# Description: Implements FVI Reliability Check using ANOVA
# ============================================================


import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))

from fvi import compute_fvi



def run_anova(df, feature, group_col='Group'):
    grouped_data = [group[feature].dropna().values for name, group in df.groupby(group_col)]
    f_stat, p_val = f_oneway(*grouped_data)
    return round(f_stat, 4), round(p_val, 4)


def generate_scenarios():
    np.random.seed(42)

    identical = pd.DataFrame({
        'Group': np.repeat(['A', 'B', 'C'], 30),
        'Feature1': np.tile(np.random.normal(5.0, 0.5, 30), 3)
    })

    one_high = pd.DataFrame({
        'Group': np.repeat(['A', 'B', 'C'], 30),
        'Feature1': np.concatenate([
            np.random.normal(5.0, 0.5, 30),
            np.random.normal(5.0, 0.5, 30),
            np.random.normal(7.0, 0.5, 30)
        ])
    })

    group_means = {'A': 5.0, 'B': 6.2, 'C': 4.3}
    mixed = pd.concat([
        pd.DataFrame({'Group': g, 'Feature1': np.random.normal(mu, 0.5, 30)})
        for g, mu in group_means.items()
    ])

    high_variance = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(2.0, 0.5, 30)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(8.0, 0.5, 30)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(14.0, 0.5, 30)}),
    ])

    unequal_sizes = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(5.0, 0.5, 10)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(5.0, 0.5, 30)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(5.0, 0.5, 50)}),
    ])

    skewed_sizes = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(5.0, 0.5, 5)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(5.0, 0.5, 50)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(5.0, 0.5, 100)}),
    ])

    large_samples = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(5.0, 0.5, 100)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(5.2, 0.5, 100)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(5.1, 0.5, 100)}),
    ])


    test_weight_proportionate = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(5.0, 0.5, 100)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(5.0, 0.5, 30)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(5.0, 0.5, 10)}),
    ])

    test_small_group_extreme = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(5.0, 0.5, 100)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(5.0, 0.5, 100)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(10.0, 0.5, 5)}),
    ])

    test_small_shift_high_weight = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(5.0, 0.5, 300)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(5.1, 0.5, 300)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(5.05, 0.5, 10)}),
    ])

    test_large_group_high_variance = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(5.0, 0.5, 100)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(5.0, 2.0, 100)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(5.0, 0.5, 100)}),
    ])

    test_two_large_one_tiny_outlier = pd.concat([
        pd.DataFrame({'Group': 'A', 'Feature1': np.random.normal(5.0, 0.5, 500)}),
        pd.DataFrame({'Group': 'B', 'Feature1': np.random.normal(5.0, 0.5, 500)}),
        pd.DataFrame({'Group': 'C', 'Feature1': np.random.normal(10.0, 0.5, 1)}),
    ])

    return {
        "Identical Means": identical,
        "One Group Higher": one_high,
        "Mixed Group Means": mixed,
        "High Variance": high_variance,
        "Unequal Sample Sizes": unequal_sizes,
        "Skewed Sample Distribution": skewed_sizes,
        "Large Sample Size with Small Group Differences": large_samples,
        "Weight Proportionate Test": test_weight_proportionate,
        "Small Group Extreme Outlier": test_small_group_extreme,
        "Small Shift High Weight": test_small_shift_high_weight,
        "Large Group High Variance": test_large_group_high_variance,
        "Two Large One Tiny Outlier": test_two_large_one_tiny_outlier
    }



def compute_unweighted_fvi(df, feature, group_by_col):
    grouped = df.groupby(group_by_col)[feature].agg(['mean', 'count']).dropna()
    means = grouped['mean']
    unweighted_mean = means.mean()
    unweighted_std = means.std()
    unweighted_cv = unweighted_std / abs(unweighted_mean) if unweighted_mean != 0 else np.nan
    return unweighted_mean, unweighted_std, unweighted_cv



st.set_page_config(layout="wide")
st.title("\U0001F9EA FVI Validation Across Scenarios")

scenarios = generate_scenarios()
all_results = []

for scenario_name, df in scenarios.items():
    st.markdown(f"## \U0001F52C Scenario: **{scenario_name}**")

    st.markdown("**\U0001F4C4 Data Preview:**")
    st.dataframe(df)

    fvi_result = compute_fvi(df, selected_features=["Feature1"], group_by_col="Group")
    fvi_summary = fvi_result.drop_duplicates(subset=["Feature", "Group_By"])

    f_stat, p_val = run_anova(df, "Feature1")

    st.markdown("**\U0001F4CA FVI Summary:**")
    st.dataframe(fvi_summary[["Feature", "FVI_Std", "FVI_CV", "Weighted_Mean"]].set_index("Feature"))

    st.markdown("**\U0001F4C8 ANOVA Result:**")
    st.markdown(f"- **F-statistic**: `{f_stat}`")
    st.markdown(f"- **p-value**: `{p_val}`")

    st.markdown("**\U0001F50D Interpretation:**")
    if fvi_summary['FVI_CV'].values[0] < 0.1 and p_val > 0.05:
        st.success(" Low variability and ANOVA confirms no significant difference.")
    elif fvi_summary['FVI_CV'].values[0] > 0.5 and p_val < 0.05:
        st.warning(" High variability and ANOVA confirms significant group differences.")
    else:
        st.info("ℹ Ambiguous result — review group distribution or increase differences for clarity.")

    fvi_export = fvi_result.copy()
    fvi_export["ANOVA_F_Statistic"] = f_stat
    fvi_export["ANOVA_p_value"] = p_val
    fvi_export["Scenario"] = scenario_name
    all_results.append(fvi_export)

    export_csv = fvi_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"⬇Download CSV for '{scenario_name}'",
        data=export_csv,
        file_name=f"fvi_anova_{scenario_name.replace(' ', '_').lower()}.csv",
        mime="text/csv"
    )

    st.markdown("---")


st.subheader(" Download Combined CSV for All Scenarios")
combined_df = pd.concat(all_results, ignore_index=True)
combined_csv = combined_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Download Combined FVI + ANOVA CSV",
    data=combined_csv,
    file_name="fvi_anova_all_scenarios.csv",
    mime="text/csv"
)

