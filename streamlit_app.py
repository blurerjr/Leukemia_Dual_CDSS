import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# =====================================================================
# 1. PAGE LAYOUT CONFIGURATION
# =====================================================================
st.set_page_config(
    page_title="Leukemia CDSS Dashboard", 
    page_icon="🩸", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling to mirror a premium medical grade UI
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #6f42c1;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .result-box-binary {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        padding: 20px;
        border-radius: 10px;
    }
    .result-box-multi {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)  # <-- FIXED: Changed from unsafe_with_html to unsafe_allow_html

# =====================================================================
# 2. RUNTIME RESOURCE PIPELINE LOADER
# =====================================================================
@st.cache_resource
def load_clinical_assets(task_path_key):
    """
    Safely unpacks model elements depending on the triggered engine route.
    """
    base_dir = f"exported_assets/{task_path_key}/"
    
    # Validation file checking to prevent app crashes on first deploy
    required_files = [
        'leukemia_rbf_svm_model.pkl', 'gene_minmax_scaler.pkl', 
        'leukemia_label_encoder.pkl', 'alo_dat_selected_genes.pkl', 
        'clinical_performance_metadata.json'
    ]
    
    for f_name in required_files:
        if not os.path.exists(os.path.join(base_dir, f_name)):
            return None
            
    with open(os.path.join(base_dir, 'leukemia_rbf_svm_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(base_dir, 'gene_minmax_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(base_dir, 'leukemia_label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(base_dir, 'alo_dat_selected_genes.pkl'), 'rb') as f:
        genes = pickle.load(f)
    with open(os.path.join(base_dir, 'clinical_performance_metadata.json'), 'r') as f:
        metrics = json.load(f)
        
    return {"model": model, "scaler": scaler, "encoder": le, "genes": genes, "metrics": metrics}

# Pre-fetch configurations from the repository paths
binary_engine = load_clinical_assets("binary_class")
multi_engine = load_clinical_assets("multi_class")

# =====================================================================
# 3. TOP TABS NAVIGATION LAYER
# =====================================================================
tab_dashboard, tab_info, tab_history = st.tabs([
    "📊 Diagnostic Workspace", 
    "🧬 ALO-DAT Model Specifications", 
    "📂 Historical Audit Trail"
])

# =====================================================================
# TAB 1: MAIN WORKSPACE
# =====================================================================
with tab_dashboard:
    st.markdown("### 🩸 Clinical Decision Support Workspace")
    st.write("Upload patient microarray records below and trigger target prediction actions directly.")
    
    # Creating layout columns split: Left Controls vs Right Presentation View
    col_control, col_display = st.columns([1, 2], gap="large")
    
    with col_control:
        st.markdown("#### 📥 Data Intake Panel")
        uploaded_file = st.file_uploader(
            "Upload Patient Microarray Expression Vector (.csv)", 
            type=['csv'],
            help="Expects expression profile across target gene probe names matrix rows."
        )
        
        st.divider()
        st.markdown("#### ⚙️ Execution Triggers")
        st.caption("No manual routing parameters required. Select your targeted clinical query option below:")
        
        # Operational Buttons Strategy
        trigger_binary = st.button("▶️ Execute Binary Prediction (AML vs Healthy)", use_container_width=True)
        trigger_multi = st.button("▶️ Execute Multi-class Prediction (Subtypes)", use_container_width=True)
        
        # Context warnings checking configuration states before letting buttons break
        if trigger_binary and binary_engine is None:
            st.error("Binary model missing inside `exported_assets/binary_class/` directory.")
        if trigger_multi and multi_engine is None:
            st.error("Multi-class model missing inside `exported_assets/multi_class/` directory.")

    with col_display:
        st.markdown("#### 🩺 System Response & Diagnostic Output")
        
        # State A: Default display view before interactions take place
        if not trigger_binary and not trigger_multi:
            st.info("💡 Ready for patient telemetry. Upload a microarray sample file and choose an execution trigger in the control panel to view results.")
            
        # State B: Handling Binary Prediction Workflow
        elif trigger_binary and uploaded_file is not None and binary_engine is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                clean_cols = [col for col in ['samples', 'type'] if col in raw_df.columns]
                patient_features = raw_df.drop(columns=clean_cols) if clean_cols else raw_df
                
                # Inference processing steps using binary configuration rules
                patient_log2 = np.log2(patient_features + 1)
                patient_scaled = binary_engine["scaler"].transform(patient_log2)
                patient_filtered = patient_scaled[:, binary_engine["genes"]['selected_indices']]
                
                prediction_idx = binary_engine["model"].predict(patient_filtered)
                result_text = binary_engine["encoder"].inverse_transform(prediction_idx)[0]
                
                # Clean Output Presentation Card
                st.markdown(f"""
                <div class="result-box-binary">
                    <h3 style='margin-top:0; color:#16a34a;'>✅ Binary Analysis Finished</h3>
                    <p style='font-size:1.1rem; color:#1f2937;'>Predicted Clinical State: <strong>{result_text}</strong></p>
                </div>
                """, unsafe_allow_html=True)  # <-- FIXED: Changed parameter name here
                
                # Show dynamic telemetry
                st.markdown("##### 🧬 Active Biomarker Profile Metrics")
                biomarker_df = pd.DataFrame({
                    'Biomarker Probe ID': binary_engine["genes"]['gene_probe_names'],
                    'Scaled Value': patient_filtered[0]
                })
                st.dataframe(biomarker_df, use_container_width=True, height=250)
                
            except Exception as e:
                st.error(f"Execution Error: Ensure column indices fit model expectations. Detailed error: {e}")
                
        # State C: Handling Multi-Class Prediction Workflow
        elif trigger_multi and uploaded_file is not None and multi_engine is not None:
            try:
                raw_df = pd.read_csv(uploaded_file)
                clean_cols = [col for col in ['samples', 'type'] if col in raw_df.columns]
                patient_features = raw_df.drop(columns=clean_cols) if clean_cols else raw_df
                
                # Inference steps using filtered multi-class subtype configuration matrices
                patient_log2 = np.log2(patient_features
