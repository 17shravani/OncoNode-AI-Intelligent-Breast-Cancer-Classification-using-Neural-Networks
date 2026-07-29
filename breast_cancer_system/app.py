import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import sklearn.datasets

# Set page config
st.set_page_config(
    page_title="Breast Cancer Diagnostics Dashboard",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich premium medical style
st.markdown("""
<style>
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #d11a5e;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Inter', sans-serif;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        border-left: 5px solid #d11a5e;
        text-align: center;
    }
    .prediction-malignant {
        background-color: #fff0f3;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #ff8093;
        color: #c9184a;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
    .prediction-benign {
        background-color: #f4fbf7;
        padding: 1.5rem;
        border-radius: 12px;
        border: 2px solid #a3e2c9;
        color: #1b4332;
        font-size: 1.5rem;
        font-weight: bold;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Cache data & model loading for speed
@st.cache_resource
def load_assets():
    scaler = joblib.load("breast_cancer_system/models/scaler.joblib")
    best_model = joblib.load("breast_cancer_system/models/best_model.joblib")
    
    with open("breast_cancer_system/models/model_metadata.json", "r") as f:
        meta = json.load(f)
        
    with open("breast_cancer_system/models/metrics.json", "r") as f:
        metrics = json.load(f)
        
    cancer = sklearn.datasets.load_breast_cancer()
    df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    df['label'] = cancer.target
    
    return scaler, best_model, meta, metrics, df, cancer.feature_names, cancer.target_names

try:
    scaler, best_model, meta, metrics_list, df, feature_names, target_names = load_assets()
except Exception as e:
    st.error(f"Error loading model assets. Have you run `train_pipeline.py` yet? Details: {e}")
    st.stop()

# Header Section
st.markdown("<div class='main-title'>Breast Cancer Diagnostics AI</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Interactive Clinician Assistant & Model Performance Dashboard</div>", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://img.icons8.com/color/96/000000/ribbon.png", width=70)
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Patient Predictor", "Dataset Analysis", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Current Best Model:** {meta['best_model_name']}")
st.sidebar.markdown(f"**Optimized Metric:** Recall (sensitivity)")
st.sidebar.markdown(f"**Recall (Malignant):** {meta['metrics']['recall']:.2%}")
st.sidebar.markdown(f"**ROC-AUC Score:** {meta['metrics']['roc_auc']:.2%}")

# --- TABS / SCREENS ---

if app_mode == "Patient Predictor":
    st.header("📋 Patient Profile Prediction")
    st.write("Enter patient tumor characteristics below to classify the breast mass as **Malignant** (Cancerous) or **Benign** (Non-cancerous).")
    
    # Preset Patient Profiles for Quick Testing
    st.subheader("💡 Load Sample Patient Data")
    sample_col, _ = st.columns([1, 2])
    profile_type = sample_col.selectbox(
        "Choose profile type to pre-fill sliders:",
        ["Manual Entry", "Typical Malignant Case", "Typical Benign Case"]
    )
    
    # Establish defaults
    defaults = {}
    if profile_type == "Typical Malignant Case":
        # Group averages for target 0 (Malignant)
        malignant_df = df[df['label'] == 0].drop(columns='label')
        defaults = malignant_df.mean().to_dict()
    elif profile_type == "Typical Benign Case":
        # Group averages for target 1 (Benign)
        benign_df = df[df['label'] == 1].drop(columns='label')
        defaults = benign_df.mean().to_dict()
    else:
        # Overall dataset means
        defaults = df.drop(columns='label').mean().to_dict()
        
    st.markdown("---")
    
    # Inputs Grouped for high usability
    st.subheader("🧬 Tumor Feature Parameters")
    
    # Form to group submission
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)
        
        # 1. Mean Parameters (Columns 0 to 9)
        with col1:
            st.markdown("### 📊 Mean Values")
            f_inputs = {}
            for feat in feature_names[:10]:
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                step_val = (max_val - min_val) / 100.0
                f_inputs[feat] = st.slider(
                    feat.replace("mean ", "").capitalize(),
                    min_value=min_val,
                    max_value=max_val,
                    value=float(defaults[feat]),
                    step=step_val
                )
                
        # 2. Standard Error Parameters (Columns 10 to 19)
        with col2:
            st.markdown("### 📐 Standard Error (SE)")
            for feat in feature_names[10:20]:
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                step_val = (max_val - min_val) / 100.0
                f_inputs[feat] = st.slider(
                    feat.replace(" error", "").capitalize() + " SE",
                    min_value=min_val,
                    max_value=max_val,
                    value=float(defaults[feat]),
                    step=step_val
                )
                
        # 3. Worst / Largest Parameters (Columns 20 to 29)
        with col3:
            st.markdown("### 🔍 Worst (Largest) Values")
            for feat in feature_names[20:]:
                min_val = float(df[feat].min())
                max_val = float(df[feat].max())
                step_val = (max_val - min_val) / 100.0
                f_inputs[feat] = st.slider(
                    feat.replace("worst ", "").capitalize() + " Worst",
                    min_value=min_val,
                    max_value=max_val,
                    value=float(defaults[feat]),
                    step=step_val
                )
                
        submit_btn = st.form_submit_button("🩺 Run Diagnostic Evaluation", use_container_width=True)

    if submit_btn or profile_type != "Manual Entry":
        # Formulate input vector
        input_vector = np.array([f_inputs[f] for f in feature_names]).reshape(1, -1)
        
        # Standardize features
        input_std = scaler.transform(input_vector)
        
        # Run prediction
        pred = best_model.predict(input_std)[0]
        probs = best_model.predict_proba(input_std)[0]
        
        st.markdown("---")
        st.subheader("🩺 Diagnostic Output")
        
        res_col, gauge_col = st.columns([1, 1])
        
        with res_col:
            st.write("#### Classification Result")
            if pred == 0:
                st.markdown("""
                <div class='prediction-malignant'>
                    ⚠️ Class: MALIGNANT (Cancerous)<br>
                    <span style='font-size: 0.95rem; font-weight: normal;'>Urgent clinical intervention recommended.</span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class='prediction-benign'>
                    ✅ Class: BENIGN (Non-Cancerous)<br>
                    <span style='font-size: 0.95rem; font-weight: normal;'>Mass shows non-cancerous features. Routine checkup advised.</span>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.write(f"**Confidence Metrics:**")
            st.write(f"- Probability of Malignancy (Class 0): **{probs[0]:.2%}**")
            st.write(f"- Probability of Benignity (Class 1): **{probs[1]:.2%}**")
            st.write(f"- Model Used: `{meta['best_model_name']}`")
            
        with gauge_col:
            st.write("#### Malignancy Risk Gauge")
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = probs[0] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Malignancy Probability (%)", 'font': {'size': 16}},
                gauge = {
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "#d11a5e"},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 40], 'color': '#d8f3dc'},
                        {'range': [40, 75], 'color': '#ffe3e0'},
                        {'range': [75, 100], 'color': '#ffccd5'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(height=260, margin=dict(t=30, b=0, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Dataset Analysis":
    st.header("📊 Dataset Exploratory Analysis")
    st.write("Visualize the Wisconsin Breast Cancer dataset to study correlations and feature splits between Benign and Malignant instances.")
    
    eda_tab1, eda_tab2, eda_tab3 = st.tabs(["2D PCA Projection", "Feature Distributions", "Correlation Matrix"])
    
    with eda_tab1:
        st.subheader("🌀 2D Principal Component Analysis (PCA) Projection")
        st.write("Reducing the 30 dimensions into 2 Principal Components to check dataset separability. Benign features cluster nicely apart from Malignant features.")
        
        # Fit PCA
        X_all = df.drop(columns='label')
        Y_all = df['label']
        X_all_std = scaler.fit_transform(X_all)
        
        pca = PCA(n_components=2)
        pcs = pca.fit_transform(X_all_std)
        pca_df = pd.DataFrame(pcs, columns=['PC 1', 'PC 2'])
        pca_df['Diagnosis'] = Y_all.map({0: 'Malignant', 1: 'Benign'})
        
        fig = px.scatter(
            pca_df, x='PC 1', y='PC 2', color='Diagnosis',
            color_discrete_map={'Malignant': '#d11a5e', 'Benign': '#2d6a4f'},
            opacity=0.8,
            title="Interactive PCA Plot (Explaining variance: {:.2%})".format(sum(pca.explained_variance_ratio_))
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with eda_tab2:
        st.subheader("📈 Feature Value Distribution")
        sel_feat = st.selectbox("Select Feature to view distribution:", feature_names)
        
        plot_df = df.copy()
        plot_df['Diagnosis'] = plot_df['label'].map({0: 'Malignant', 1: 'Benign'})
        
        fig = px.histogram(
            plot_df, x=sel_feat, color='Diagnosis',
            color_discrete_map={'Malignant': '#d11a5e', 'Benign': '#2d6a4f'},
            marginal="box", barmode="overlay",
            title=f"Distribution of {sel_feat.capitalize()}"
        )
        st.plotly_chart(fig, use_container_width=True)
        
    with eda_tab3:
        st.subheader("🔗 Feature Correlation Heatmap")
        st.write("Shows correlation between mean features. Higher correlation indicates stronger relation (e.g. area and perimeter).")
        
        mean_feats = [f for f in feature_names if 'mean' in f] + ['label']
        corr_matrix = df[mean_feats].corr()
        
        fig = px.imshow(
            corr_matrix, text_auto=".2f",
            color_continuous_scale="RdBu_r",
            title="Correlation Heatmap (Means & Target)"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.header("🏆 Model Comparison & Interpretation")
    st.write("Compare metric results across all tuned algorithms and view model interpretation charts.")
    
    # Leaderboard metrics
    st.subheader("Leaderboard Table")
    metrics_df = pd.DataFrame(metrics_list).sort_values(by='recall', ascending=False)
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, subset=['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'pr_auc'], color="#e9d8e4"),
        use_container_width=True
    )
    
    # Bar Chart comparison
    st.write("#### Performance Metric Chart")
    plot_metrics_df = pd.melt(metrics_df, id_vars=['model_name'], value_vars=['accuracy', 'recall', 'roc_auc'], var_name='Metric', value_name='Score')
    fig = px.bar(
        plot_metrics_df, x='model_name', y='Score', color='Metric',
        barmode='group', color_discrete_sequence=['#457b9d', '#e63946', '#a8dadc'],
        title="Side-by-Side Model Comparison (Test Set)"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Plot Image Assets
    st.subheader("🔍 Model Validation Visualizations")
    col_cm, col_roc = st.columns(2)
    
    with col_cm:
        st.write("#### Confusion Matrix (Selected Model)")
        # Show best model's confusion matrix
        safe_name = meta['best_model_name'].lower().replace(' ', '_').replace('(', '').replace(')', '')
        cm_file = f"breast_cancer_system/plots/confusion_matrix_{safe_name}.png"
        if os.path.exists(cm_file):
            st.image(cm_file, caption=f"Confusion Matrix for the top performing {meta['best_model_name']} model.")
            
    with col_roc:
        st.write("#### ROC Curves comparison")
        roc_file = "breast_cancer_system/plots/roc_curve_comparison.png"
        if os.path.exists(roc_file):
            st.image(roc_file, caption="ROC-Curves comparing true positive vs false positive trade-offs for all models.")
            
    st.markdown("---")
    
    st.subheader("💡 Model Interpretability via SHAP")
    st.write("Global feature importances explaining how features influence the models. High SHAP values push predictions towards Benign, while low values point towards Malignant.")
    
    col_shap_xgb, col_shap_rf = st.columns(2)
    
    with col_shap_xgb:
        shap_xgb = "breast_cancer_system/plots/shap_summary_xgboost.png"
        if os.path.exists(shap_xgb):
            st.image(shap_xgb, caption="SHAP Summary Plot for tuned XGBoost classifier.")
            
    with col_shap_rf:
        shap_rf = "breast_cancer_system/plots/shap_summary_random_forest.png"
        if os.path.exists(shap_rf):
            st.image(shap_rf, caption="SHAP Summary Plot for tuned Random Forest classifier.")
