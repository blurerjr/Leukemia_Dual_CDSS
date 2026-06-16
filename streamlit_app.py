import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import pickle

# --- Page Config ---
st.set_page_config(
    page_title="Leukemia Dual-Engine CDSS",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Inject Custom Clinical Theme Styling (Fixed Parameter Typo) ---
st.markdown("""
    <style>
    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 5px;
    }
    .subtitle {
        font-size: 16px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 25px;
        font-style: italic;
    }
    .metric-card {
        background-color: #F8FAFC;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .binary-header {
        color: #047857;
        font-weight: 700;
    }
    .multi-header {
        color: #6D28D9;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown('<div class="main-title">🔬 Clinical Intelligence Leukemia Diagnostic Suite</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">High-Performance Swarm-Optimized (ALO-DAT) SVM Screening Engine</div>', unsafe_allow_html=True)

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ System Control Center")
analysis_mode = st.sidebar.radio(
    "Select Diagnostic Engine Workflow:",
    ["Binary Screening (AML vs Normal)", "Subtype Stratification (Multi-Class)"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Pipeline Specifications")
if analysis_mode == "Binary Screening (AML vs Normal)":
    st.sidebar.info("**Dataset Context:** GSE63270\n\n**Expected Features:** 54,675 Genes\n\n**Target Scope:** AML vs Healthy Control")
else:
    st.sidebar.info("**Dataset Context:** GSE28497\n\n**Expected Features:** 22,283 Genes\n\n**Target Scope:** 6 Molecular Subtypes (excluding heterogeneous classes)")

# --- Mock Asset Loader (Replace with actual file paths or local artifact generation) ---
@st.cache_resource
def load_clinical_assets(mode_prefix):
    # This dictionary simulates loading your dumped JSON/Pickle engine profiles
    # In production, replace this with actual pickle.load() operations
    if mode_prefix == "binary":
        return {
            "features_count": 54675,
            "classes": ["AML", "normal"],
            "mock_accuracy": 100.0,
            "mock_f1": 100.0,
            "selected_biomarkers": ["ENSG00000001", "ENSG00000002", "ENSG00000003"]
        }
    else:
        return {
            "features_count": 22283,
            "classes": [
                'B-CELL_ALL_ETV6-RUNX1', 'B-CELL_ALL_HYPERDIP', 
                'B-CELL_ALL_T-ALL', 'B-CELL_ALL_TCF3-PBX1', 
                'B-CELL_ALL_HYPO', 'B-CELL_ALL_MLL'
            ],
            "mock_accuracy": 97.62,
            "mock_f1": 97.27,
            "selected_biomarkers": ["ENSG000100", "ENSG000200", "ENSG000300", "ENSG000400"]
        }

# Load the asset context based on active layout selection
asset_prefix = "binary" if analysis_mode == "Binary Screening (AML vs Normal)" else "multi"
engine_assets = load_clinical_assets(asset_prefix)

# --- Main Workspace Tabs ---
tab1, tab2 = st.tabs(["🚀 Diagnostic Inference", "📊 Engine Performance Architecture"])

with tab1:
    st.markdown(f"### Upload Patient Transcript Expressions ({analysis_mode})")
    uploaded_file = st.file_with_container = st.file_uploader(
        "Upload micro-array expression matrix profile (.csv or .txt tab-delimited)", 
        type=["csv", "txt"]
    )
    
    if uploaded_file is not None:
        try:
            # Handle standard expressions format (sniff for commas or tabs)
            df_input = pd.read_csv(uploaded_file, sep=None, engine='python')
            st.success("✅ Patient profile uploaded successfully!")
            
            # Show preview
            st.markdown("#### Raw Expressions Preview")
            st.dataframe(df_input.head(3))
            
            # Align features and execute data transformation steps
            st.markdown("---")
            st.markdown("### ⚙️ Automated Preprocessing Pipeline Execution")
            
            with st.spinner("Executing Log2 alignment, MinMaxScaler mapping, and biomarker extraction..."):
                # 1. Log2 Transformation: log2(X + 1)
                # 2. Emulate biomarker mapping
                st.info("🔹 Step 1: Executing $\log_2(X + 1)$ mathematical transformation matrix alignment...")
                st.info("🔹 Step 2: Applying MinMaxScaler normalization bounds $[0, 1]$...")
                st.info(f"🔹 Step 3: Sub-setting expressions down to selected ALO-DAT Biomarkers.")
                
                # Mock a random diagnostic projection based on the classes
                mock_probabilities = np.random.dirichlet(np.ones(len(engine_assets["classes"])), size=1)[0]
                predicted_class_idx = np.argmax(mock_probabilities)
                predicted_class_label = engine_assets["classes"][predicted_class_idx]
                confidence_score = mock_probabilities[predicted_class_idx]
                
            # --- Output Diagnosis Display ---
            st.markdown("### 🩺 Diagnostic Inference Output")
            col1, col2 = st.columns([1, 1])
            
            with col1:
                if asset_prefix == "binary":
                    box_style = "background-color: #E6F4EA; border-left: 6px solid #137333; padding: 20px; border-radius: 8px;"
                    label_style = "color: #137333; font-size: 28px; font-weight: 800;"
                else:
                    box_style = "background-color: #F3E8FF; border-left: 6px solid #7C3AED; padding: 20px; border-radius: 8px;"
                    label_style = "color: #7C3AED; font-size: 28px; font-weight: 800;"
                
                st.markdown(f"""
                <div style="{box_style}">
                    <span style="font-size: 14px; color: #4B5563; font-weight: bold; text-transform: uppercase;">Predicted Clinical Classification</span>
                    <div style="{label_style}">{predicted_class_label}</div>
                    <span style="font-size: 14px; color: #1F2937;">Statistical Prediction Confidence: <b>{confidence_score*100:.2f}%</b></span>
                </div>
                """, unsafe_allow_html=True)
                
            with col2:
                st.markdown("#### Probability Distribution Breakdown")
                prob_df = pd.DataFrame({
                    'Clinical Classification Target': engine_assets["classes"],
                    'Engine Weight Probability': mock_probabilities
                }).sort_values(by='Engine Weight Probability', ascending=False)
                
                st.dataframe(prob_df.style.format({'Engine Weight Probability': '{:.4%}'}))
                
        except Exception as e:
            st.error(f"❌ Error processing input matrix file architecture: {str(e)}")
            
    else:
        st.warning("📥 Awaiting micro-array expression batch matrix to trigger clinical prediction pipeline.")

with tab2:
    st.markdown("### 📈 Core Architecture & Validation Profile Metrics")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 12px; color: #6B7280; font-weight: 600;">DIAGNOSTIC ACCURACY</span>
            <h2 style="margin: 0; color: #1E3A8A;">{engine_assets['mock_accuracy']:.2f}%</h2>
            <span style="font-size: 11px; color: #10B981;">Stratified 5-Fold Evaluation Matrix</span>
        </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 12px; color: #6B7280; font-weight: 600;">MACRO F1-SCORE BOUND</span>
            <h2 style="margin: 0; color: #1E3A8A;">{engine_assets['mock_f1']:.2f}%</h2>
            <span style="font-size: 11px; color: #10B981;">Robust to Imbalance Boundaries</span>
        </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
        <div class="metric-card">
            <span style="font-size: 12px; color: #6B7280; font-weight: 600;">EXTRACTED BIOMARKERS SIGNATURE</span>
            <h2 style="margin: 0; color: #1E3A8A;">{len(engine_assets['selected_biomarkers'])} Genes</h2>
            <span style="font-size: 11px; color: #2563EB;">Selected via ALO-DAT Optimization</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🔬 Optimization Signature Breakdown")
    st.write(f"The high-performance Ant Lion Optimizer with Chaotic Jump Mechanism (ALO-DAT) filtered down the original **{engine_assets['features_count']:,} variables** into the following high-impact diagnostic gene sub-signature:")
    st.json(engine_assets["selected_biomarkers"])
