import streamlit as st
import shap
import joblib
import matplotlib.pyplot as plt
import os

# --- Streamlit Page Setup ---
st.set_page_config(
    page_title="Model Validation & Interpretability Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔍 Model Interpretability Dashboard")
st.markdown("""
This dashboard displays **SHAP-based feature importance** results from the Home Credit Default Risk model.
""")

# --- Load Data ---
@st.cache_resource
def load_shap_results():
    output_dir = "Steven_Maxwell_Portfolio/projects/model_validation_practice/model_validation_project/outputs/shap_results"
    model = joblib.load(os.path.join(output_dir, "xgb_model.pkl"))
    shap_values = joblib.load(os.path.join(output_dir, "shap_values.pkl"))
    X_test = joblib.load(os.path.join(output_dir, "X_test.pkl"))
    return model, shap_values, X_test

try:
    model, shap_values, X_test = load_shap_results()
    st.success("Successfully loaded model and SHAP values.")
except Exception as e:
    st.error("Failed to load SHAP data. Please run `shap_interpretability.py` first.")
    st.exception(e)
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("Navigation")
plot_type = st.sidebar.radio(
    "Choose a visualization:",
    ["Global Importance (Bar Chart)", "Feature Impact (Beeswarm)", "Feature Dependence (Scatter)"]
)

# --- Plot: Global Feature Importance ---
if plot_type == "Global Importance (Bar Chart)":
    st.subheader("Global Feature Importance (Mean|SHAP Values|)")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

# --- Plot: Beeswarm Overview ---
elif plot_type == "Feature Impact (Beeswarm)":
    st.subheader("Feature Impact Overview (Beeswarm Plot)")
    st.markdown("Each point represents a SHAP value for a feature across all samples.")
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)

# --- Plot: Dependence ---
elif plot_type == "Feature Dependence (Scatter)":
    st.subheader("Feature Dependence Plot")
    st.markdown("Visualize how SHAP values vary with a specific feature.")
    selected_feature = st.selectbox("Select a feature to visualize:", X_test.columns)
    shap.dependence_plot(selected_feature, shap_values, X_test, show=False)
    st.pyplot(plt.gcf())
    plt.clf()

# --- Footer ---
st.markdown("---")
st.caption("Built using Streamlit and SHAP for model validation interpretability.")
