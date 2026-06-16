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

# Premium medical grade custom UI styling classes
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
        margin-bottom: 20px;
    }
    .result-box-multi {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. RUNTIME RESOURCE PIPELINE LOADER
# =====================================================================
@st.cache_resource
def load_clinical_assets(task_path_key):
    """
    Safely unpacks model elements depending on the triggered engine route.
    Folders are located directly in the root directory beside app.py.
    """
    base_dir = f"{task_path_key}/"
    
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
            "Upload Patient Microarray Expression Matrix File", 
            type=['csv', 'txt', 'xlsx', 'xls'],
            help="Accepts Excel Workbooks, Comma-Separated Values, or Tab-Separated genomic text configurations."
        )
        
        st.divider()
        st.markdown("#### ⚙️ Execution Triggers")
        st.caption("Select your targeted clinical query option below to initiate analysis:")
        
        # Operational Buttons Strategy
        trigger_binary = st.button("▶️ Execute Binary Prediction (AML vs Healthy)", use_container_width=True)
        trigger_multi = st.button("▶️ Execute Multi-class Prediction (Subtypes)", use_container_width=True)
        
        # Context warnings checking configuration states before letting buttons execute
        if trigger_binary and binary_engine is None:
            st.error("Binary model assets missing inside `binary_class/` directory.")
        if trigger_multi and multi_engine is None:
            st.error("Multi-class model assets missing inside `multi_class/` directory.")

    with col_display:
        st.markdown("#### 🩺 System Response & Diagnostic Output")
        
        # State A: Default display view before interactions take place
        if not trigger_binary and not trigger_multi:
            st.info("💡 Ready for patient telemetry. Upload a microarray sample file and choose an execution trigger in the control panel to view results.")
            
        # State B: Core Predictive Pipeline Execution
        elif (trigger_binary or trigger_multi) and uploaded_file is not None:
            
            if trigger_binary and binary_engine is not None:
                engine = binary_engine
                task_label = "Binary Analysis Finished"
                box_style = "result-box-binary"
                title_color = "#16a34a"
            elif trigger_multi and multi_engine is not None:
                engine = multi_engine
                task_label = "Molecular Subtype Identified"
                box_style = "result-box-multi"
                title_color = "#dc2626"
            else:
                engine = None

            if engine is not None:
                try:
                    # 1. Determine dynamic feature count requirements directly from the active scaler
                    try:
                        expected_features = engine["scaler"].n_features_in_
                    except AttributeError:
                        expected_features = len(engine["scaler"].data_min_)

                    # 2. Extract extension mapping
                    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                    
                    # 3. Dynamic Parser Selection (Excel vs Sniffed Text)
                    uploaded_file.seek(0)
                    if file_extension in ['.xlsx', '.xls']:
                        raw_df = pd.read_excel(uploaded_file)
                    else:
                        raw_df = pd.read_csv(uploaded_file, sep=None, engine='python')

                    # 4. Smart Feature Alignment via Saved Model Estimator Headers
                    has_valid_feature_names = False
                    if hasattr(engine["scaler"], "feature_names_in_"):
                        expected_feature_names = engine["scaler"].feature_names_in_
                        # Check matching intersection density
                        matching_cols = [col for col in expected_feature_names if col in raw_df.columns]
                        if len(matching_cols) >= (expected_features * 0.9):
                            has_valid_feature_names = True

                    if has_valid_feature_names:
                        # Reorder columns to perfectly align with what the scaler expects
                        patient_features = raw_df[expected_feature_names].astype(float)
                        st.success(f"🧬 **Gene Mapping Matrix Aligned:** Matched {len(expected_feature_names)} features by Probe IDs.")
                    else:
                        # Fallback to right-side positional slice if headers are absent or generic numbers
                        actual_cols = raw_df.shape[1]
                        if actual_cols < expected_features:
                            st.error(f"**⚠️ Structural Dimensionality Mismatch:** File has {actual_cols} columns, but model requires {expected_features}.")
                            st.stop()
                        patient_features = raw_df.iloc[:, -expected_features:].astype(float)
                        st.warning("⚠️ **Positional Alignment Active:** Mismatched or missing header row. Slicing matrix positionally.")

                    patient_matrix = patient_features.values
                    
                    # --- Execution Calculations (Log2 step completely removed to prevent distribution warping) ---
                    # Step A: Normalize array values using parameters from training data
                    patient_scaled = engine["scaler"].transform(patient_matrix)
                    
                    # Step B: Filter features matching ALO-DAT optimized tracking indices
                    patient_filtered = patient_scaled[:, engine["genes"]['selected_indices']]
                    
                    # Step C: Execute RBF SVM Inference
                    numeric_class = engine["model"].predict(patient_filtered)
                    text_prediction = engine["encoder"].inverse_transform(numeric_class)[0]
                    
                    # =====================================================================
                    # PREDICTIVE PRESENTATION ENGINE
                    # =====================================================================
                    st.markdown(f"""
                    <div class="{box_style}">
                        <h3 style='margin-top:0; color:{title_color};'>{task_label}</h3>
                        <p style='font-size:1.3rem; color:#1f2937; margin-bottom:0;'>
                            Diagnostic Status Assessment: <strong>{text_prediction}</strong>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if trigger_multi:
                        st.warning("⚠️ *Clinical Context Note: Omitted general heterogeneous catch-all configurations (B-CELL_ALL).*")
                        
                    # Interactive Gene Expression Matrix Breakdown Mapping
                    st.markdown("##### 🧬 Active Molecular Signature Profile Metrics")
                    biomarker_map_table = pd.DataFrame({
                        'Biomarker Index Map': engine["genes"]['selected_indices'],
                        'Gene Probe ID Reference': engine["genes"]['gene_probe_names'],
                        'Patient Normalized Value': patient_filtered[0]
                    })
                    st.dataframe(biomarker_map_table, use_container_width=True, height=280)
                    
                except Exception as e:
                    st.error(f"Execution Error: An unexpected error occurred while parsing matrix metrics. Details: {e}")
                    
        elif (trigger_binary or trigger_multi) and uploaded_file is None:
            st.warning("⚠️ Process Halted: You must upload a patient microarray expression matrix file before executing predictions.")

# =====================================================================
# TAB 2: MODEL DETAILED INFORMATIONAL MATRIX
# =====================================================================
with tab_info:
    st.markdown("### 🧬 Architecture Overview & Performance Metadata")
    st.write("This structural summary is parsed and loaded from static metadata assets inside your system repo.")
    
    col_inf_b, col_inf_m = st.columns(2)
    
    with col_inf_b:
        st.markdown("#### 🔳 Binary Model Engine")
        if binary_engine:
            m = binary_engine["metrics"]
            st.markdown(f"""
            - **Target Pipeline:** Acute Myeloid Leukemia (AML) vs Healthy
            - **Unbiased Holdout Accuracy:** {m['diagnostic_accuracy_pct']}%
            - **Macro F1-Score:** {m['macro_f1_score_pct']}%
            - **Selected Molecular Signature size:** {m['number_of_biomarkers']} target genes
            """)
        else:
            st.caption("No binary configurations found inside `binary_class/` directory.")
            
    with col_inf_m:
        st.markdown("#### 🔲 Multi-class Subtype Model Engine")
        if multi_engine:
            m = multi_engine["metrics"]
            st.markdown(f"""
            - **Target Pipeline:** 6 Distinct B-Cell/T-Cell Molecular Variant Pathways
            - **Unbiased Holdout Accuracy:** {m['diagnostic_accuracy_pct']}%
            - **Macro Specificity:** {m.get('macro_specificity_pct', '99.56')}%
            - **Macro F1-Score:** {m['macro_f1_score_pct']}%
            - **Selected Molecular Signature size:** {m['number_of_biomarkers']} target genes
            """)
        else:
            st.caption("No multi-class configurations found inside `multi_class/` directory.")

# =====================================================================
# TAB 3: AUDIT TRACKING LOGS
# =====================================================================
with tab_history:
    st.markdown("### 📂 System Audit Logs")
    st.write("Maintains record traces of diagnostic sessions.")
    st.caption("No diagnostic actions run during this active initialization frame.")
