import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# =====================================================================
# 1. APPLICATION VIEW DESIGN & WINDOW LAYOUT
# =====================================================================
st.set_page_config(page_title="Leukemia Dual-Engine CDSS", page_icon="🔬", layout="wide")

st.title("🔬 Clinical Decision Support System (CDSS) for Leukemia Prediction")
st.markdown("""
This system provides diagnostic execution for two clear clinical workflows:
1. **Binary Framework:** General classification rules distinguishing healthy samples from malignant conditions.
2. **Multi-class Subtype Blueprint:** Precise differentiation across 6 specific chromosomal abnormalities.
""")

# =====================================================================
# 2. RUNTIME RESOURCE MANAGEMENT (CACHED LOADER)
# =====================================================================
@st.cache_resource
def load_task_assets(task_type):
    """
    Dynamically maps path nodes based on task selections.
    """
    if task_type == "Multi-class Classification":
        folder = "multi_class/"
    else:
        folder = "binary_class/"
        
    with open(os.path.join(folder, 'leukemia_rbf_svm_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(folder, 'gene_minmax_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(folder, 'leukemia_label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(folder, 'alo_dat_selected_genes.pkl'), 'rb') as f:
        genes = pickle.load(f)
    with open(os.path.join(folder, 'clinical_performance_metadata.json'), 'r') as f:
        metrics = json.load(f)
        
    return model, scaler, le, genes, metrics

# =====================================================================
# 3. INTERACTIVE CONTROL PANEL (SIDEBAR)
# =====================================================================
with st.sidebar:
    st.header("⚙️ Configuration Engine")
    
    # Task Mode Selection
    selected_task = st.selectbox(
        "Choose Classification Task Mode:",
        ["Multi-class Classification", "Binary Classification"]
    )
    
    st.divider()
    
    # Load assets matching the current selection
    try:
        svm_model, scaler, encoder, gene_payload, clinical_metrics = load_task_assets(selected_task)
        assets_loaded = True
    except FileNotFoundError as e:
        st.error(f"Asset loading failed. Verify files exist inside: `exported_assets/` directories.")
        assets_loaded = False
        
    if assets_loaded:
        st.header("📋 Core Model Validation Stats")
        st.caption(f"Unbiased Holdout Performance Metrics for {selected_task}:")
        
        # Displaying values pulled dynamically from metadata JSON logs
        st.metric(label="Diagnostic Accuracy", value=f"{clinical_metrics['diagnostic_accuracy_pct']}%")
        st.metric(label="Macro F1-Score", value=f"{clinical_metrics['macro_f1_score_pct']}%")
        st.write(f"**Isolated Biomarkers:** {clinical_metrics['number_of_biomarkers']} probe ids")
        
        st.divider()
        st.subheader("📥 Upload Patient Expression Matrix")
        uploaded_file = st.file_uploader("Choose patient microarray file (.csv)", type=['csv'])

# =====================================================================
# 4. MAIN INFERENCE COMPILING ROUTINE
# =====================================================================
if assets_loaded and uploaded_file is not None:
    # Read clinical matrix
    raw_df = pd.read_csv(uploaded_file)
    
    # Clean system inputs from label or index columns if included by users
    clean_cols = [col for col in ['samples', 'type'] if col in raw_df.columns]
    patient_features = raw_df.drop(columns=clean_cols) if clean_cols else raw_df
    
    st.header("🔬 Processing Pipeline Execution Logs")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Initial Dimensionality Status:** {patient_features.shape[1]} raw genes captured.")
    
    # --- Strict Mathematical Transformation Flow (Aligning to exact notebook boundaries) ---
    try:
        # Step A: Variance stabilization via natural shifting boundaries
        patient_log2 = np.log2(patient_features + 1)
        
        # Step B: Project using parameters learned exclusively from training vectors
        patient_scaled = scaler.transform(patient_log2)
        
        # Step C: Exact index slicing based on ALO-DAT metaheuristic outcomes
        # Eliminates the need for any complex PCA combinations
        patient_filtered = patient_scaled[:, gene_payload['selected_indices']]
        
        with col2:
            st.success(f"**ALO-DAT Active Signature:** Extracted specified {patient_filtered.shape[1]} biomarkers successfully.")
            
        # =====================================================================
        # 5. ML PREDICTION AND REPORTING DISPLAY
        # =====================================================================
        st.divider()
        st.subheader("🩺 Diagnostic Output Analysis")
        
        # Predict using optimized hyperparameter grid variables
        numeric_class = svm_model.predict(patient_filtered)
        text_prediction = encoder.inverse_transform(numeric_class)[0]
        
        # Format layout for final delivery
        st.error(f"### Diagnostic Assessment: **{text_prediction}**")
        
        # Add clinical alert warnings for Multi-class context anomalies
        if selected_task == "Multi-class Classification":
            st.warning("""
            ⚠️ *Note: This output represents identification among the specific molecular subtypes: 
            ETV6-RUNX1, HYPERDIP, T-ALL, TCF3-PBX1, HYPO, or MLL. General catch-all B-CELL_ALL profiles are omitted.*
            """)
            
        # Expandable panel plotting the detailed biological breakdown table
        with st.expander("🔬 Map Diagnostic Gene Signature Profiles"):
            biomarker_map_table = pd.DataFrame({
                'Biomarker Index Map': gene_payload['selected_indices'],
                'Gene Probe ID Reference': gene_payload['gene_probe_names'],
                'Patient Normalized Value': patient_filtered[0]
            })
            st.dataframe(biomarker_map_table, use_container_width=True, hide_index=True)
            
    except ValueError as val_err:
        st.error("""
        **Dimensionality Shape Mismatch Error:** The uploaded CSV does not align with the 22,283 genes expected 
        by this model structure. Please verify patient probe matching.
        """)
else:
    if assets_loaded:
        st.info("👈 Please select your diagnostic target module and upload a patient microarray configuration file via the sidebar control board to process predictions.")
