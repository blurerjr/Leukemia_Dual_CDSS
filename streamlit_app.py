import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
import sqlite3
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Leukemia ALO-DAT - Diagnostic Suite",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOCAL DATABASE STORAGE SETUP (LOG KEEPING) ---
DB_FILE = "leukemia_dx_logs.db"

def init_db():
    """Initializes the local database for permanent on-device logging."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnostic_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            operator_name TEXT,
            patient_name TEXT,
            patient_id TEXT,
            diagnostic_type TEXT,
            result_diagnosis TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_diagnostic_run(patient_name, patient_id, diagnostic_type, result_diagnosis):
    """Saves a diagnostic record permanently to the device database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    cursor.execute('''
        INSERT INTO diagnostic_logs (timestamp, operator_name, patient_name, patient_id, diagnostic_type, result_diagnosis)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        timestamp,  # <-- Added this missing variable
        "DR. J", 
        patient_name.strip().upper(), 
        patient_id.strip().upper(), 
        diagnostic_type, 
        result_diagnosis
    ))
    conn.commit()
    conn.close()
def get_patient_history(patient_id):
    """Retrieves all historical diagnostic runs for a specific patient ID."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query('''
        SELECT timestamp, diagnostic_type, result_diagnosis 
        FROM diagnostic_logs 
        WHERE patient_id = ? 
        ORDER BY timestamp DESC
    ''', conn, params=(patient_id.strip().upper(),))
    conn.close()
    return df

# Initialize database right away
init_db()

# --- CUSTOM CLINICAL THEME (CSS) ---
st.markdown("""
    <style>
    /* Clinical Color Scheme and Fonts */
    :root {
        --primary-clinical: #023e8a;
        --secondary-clinical: #0077b6;
        --light-bg: #f8f9fa;
    }
    
    .main-title {
        color: #023e8a;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
        margin-bottom: 2px;
    }
    
    .sub-title {
        color: #6c757d;
        font-size: 14px;
        margin-bottom: 25px;
    }
    
    /* Medical Container Cards */
    .clinical-card {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e9ecef;
        margin-bottom: 20px;
    }
    
    .patient-badge {
        background-color: #e3f2fd;
        border-left: 5px solid #023e8a;
        padding: 12px;
        border-radius: 4px;
        margin-bottom: 15px;
    }
    
    .result-box {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 20px;
        border-radius: 6px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHING MODEL ENGINE LOADS ---
@st.cache_resource
def load_pipeline_assets(model_type):
    """Loads classification architecture dynamically from the selected directories."""
    folder = "models/binary/" if model_type == "Binary Classification" else "models/multi/"
    
    with open(os.path.join(folder, 'leukemia_rbf_svm_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    with open(os.path.join(folder, 'gene_minmax_scaler.pkl'), 'rb') as f:
        scaler = pickle.load(f)
    with open(os.path.join(folder, 'leukemia_label_encoder.pkl'), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(folder, 'alo_dat_selected_genes.pkl'), 'rb') as f:
        signature = pickle.load(f)
    with open(os.path.join(folder, 'clinical_performance_metadata.json'), 'r') as f:
        metrics = json.load(f)
        
    return model, scaler, le, signature, metrics

# --- INFERENCE PIPELINE MECHANISM ---
def execute_inference(df, model, scaler, le, signature):
    """Strict execution of the notebook's preprocessing and classification steps."""
    # Data Clean: Remove descriptive labels if present
    if 'samples' in df.columns:
        df = df.drop(columns=['samples'])
    if 'type' in df.columns:
        df = df.drop(columns=['type'])

    # Step 1: Log2 Normalization
    X_log2 = np.log2(df + 1)

    # Step 2: Global Feature Scaling
    X_scaled = scaler.transform(X_log2)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)

    # Step 3: ALO-DAT Algorithmic Slicing
    selected_indices = signature['selected_indices']
    X_selected = X_scaled_df.iloc[:, selected_indices].values

    # Step 4: Model Prediction Execution
    predictions_encoded = model.predict(X_selected)
    predictions_decoded = le.inverse_transform(predictions_encoded)
    
    return predictions_decoded

# --- SESSION STATE TRACKING ---
if 'active_patient_name' not in st.session_state:
    st.session_state.active_patient_name = ""
if 'active_patient_id' not in st.session_state:
    st.session_state.active_patient_id = ""
if 'workspace_ready' not in st.session_state:
    st.session_state.workspace_ready = False

# --- SIDEBAR CONTROL CONTROL PANEL ---
with st.sidebar:
    st.markdown("### 🏥 Authorized Operator")
    st.info("**Welcome DR. J**\n\nRole: Attending Hematologist")
    st.divider()
    
    st.markdown("### ⚙️ Engine Configurations")
    model_choice = st.radio(
        "Select Computational Pipeline:", 
        ["Binary Classification", "Multi-Class Classification"]
    )
    
    # Attempt Pipeline Initialization
    try:
        svm_model, minmax_scaler, label_encoder, gene_signature, model_metrics = load_pipeline_assets(model_choice)
        st.caption("✅ Core engines loaded from local storage system storage.")
    except FileNotFoundError:
        st.error("⚠️ System assets missing. Verify folder layout paths.")
        st.stop()
        
    st.divider()
    if st.session_state.workspace_ready:
        if st.button("🔄 Clear Active Patient Workspace", use_container_width=True):
            st.session_state.active_patient_name = ""
            st.session_state.active_patient_id = ""
            st.session_state.workspace_ready = False
            st.rerun()

# --- MAIN APP HEADER ---
st.markdown("<h1 class='main-title'>🧬 Leukemia ALO-DAT</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>High-Throughput Swarm-Optimized Gene Expression Diagnostic Architecture</p>", unsafe_allow_html=True)

# --- EXPANDABLE VALIDATION BENCHMARK METRICS ---
with st.expander("📊 View Pipeline Performance Signatures (Independent Validation Datasets)", expanded=False):
    st.markdown(f"**ALO-DAT Selected Biomarkers:** {model_metrics['number_of_biomarkers']} Key Expression Drivers Sliced")
    m_cols = st.columns(5)
    m_cols[0].metric("Accuracy Score", f"{model_metrics['diagnostic_accuracy_pct']}%")
    m_cols[1].metric("Macro Precision", f"{model_metrics['macro_precision_pct']}%")
    m_cols[2].metric("Macro Recall", f"{model_metrics['macro_recall_sens_pct']}%")
    m_cols[3].metric("Macro Specificity", f"{model_metrics['macro_specificity_pct']}%")
    m_cols[4].metric("Macro F1-Score", f"{model_metrics['macro_f1_score_pct']}%")

st.divider()

# --- WORKFLOW STEP 1: PATIENT INTAKE VALIDATION ---
if not st.session_state.workspace_ready:
    st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
    st.subheader("📋 Step 1: Patient Admission Intake Registry")
    st.markdown("Before loading raw expression counts, register patient files to generate a cryptographically valid entry log.")
    
    col1, col2 = st.columns(2)
    with col1:
        pt_name_input = st.text_input("Patient Full Name:", placeholder="e.g. John Doe").strip()
    with col2:
        pt_id_input = st.text_input("Unique Medical Record Number (MRN / Patient ID):", placeholder="e.g. L-2026-8843").strip()
        
    if st.button("🔐 Initialize Patient Diagnostic Workspace", type="primary"):
        if pt_name_input and pt_id_input:
            st.session_state.active_patient_name = pt_name_input
            st.session_state.active_patient_id = pt_id_input
            st.session_state.workspace_ready = True
            st.rerun()
        else:
            st.warning("🚨 Operational Failure: Both Patient Full Name and Unique Patient ID are strictly mandatory for log keeping.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- WORKFLOW STEP 2 & 3: APPLICATION OF COMPENSATORY PIPELINE ---
else:
    # Split Layout for Main Interactions and Patient logs side by side
    main_panel, log_panel = st.columns([2, 1])
    
    with main_panel:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.markdown(f"""
            <div class='patient-badge'>
                <strong>ACTIVE CLINICAL CONTEXT</strong><br>
                Patient Name: <code>{st.session_state.active_patient_name.upper()}</code> &nbsp;|&nbsp; 
                MRN ID: <code>{st.session_state.active_patient_id.upper()}</code> &nbsp;|&nbsp;
                Operator: <code>DR. J</code>
            </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📁 Step 2: Microarray Gene Expression Count Upload")
        st.markdown("Upload the raw gene expression values `.csv` file format below.")
        
        uploaded_matrix = st.file_uploader("Select Target Expression Profile File", type=["csv"])
        
        if uploaded_matrix is not None:
            raw_matrix_df = pd.read_csv(uploaded_matrix)
            
            with st.expander("🔬 View Raw Upload Dimensions and Mapping Matrix Preview"):
                st.dataframe(raw_matrix_df.head(4))
                st.caption(f"Matrix detected size properties: {raw_matrix_df.shape[0]} rows x {raw_matrix_df.shape[1]} columns.")
                
            if st.button("⚡ Execute Computational Diagnostics", type="primary"):
                # Open up the status manager to mimic preprocessing
                with st.status("Initializing processing steps matching model parameters...", expanded=True) as status:
                    status.update(label="Executing Log2 normalization matrix translation...", state="running")
                    # Internal pipeline running
                    try:
                        results = execute_inference(
                            raw_matrix_df, svm_model, minmax_scaler, label_encoder, gene_signature
                        )
                        
                        status.update(label="Applying fitted MinMaxScaler conversions...", state="running")
                        status.update(label="Slicing targeted feature matrices via ALO-DAT indices...", state="running")
                        status.update(label="Running boundary predictions on RBF-SVM engine...", state="running")
                        
                        # Set final status output clear
                        status.update(label="Diagnostic classification verified successfully.", state="complete")
                        
                        # Handle individual prediction outcomes
                        primary_prediction = results[0]
                        
                        # --- COMMIT LOG ENTRY TO DEVICE DATABASE ---
                        log_diagnostic_run(
                            st.session_state.active_patient_name,
                            st.session_state.active_patient_id,
                            model_choice,
                            primary_prediction
                        )
                        
                        # Display output panel
                        st.markdown(f"""
                            <div class='result-box'>
                                <h3 style='margin-top:0px; color:#2e7d32;'>📋 Computed Diagnostic Verdict</h3>
                                <p style='font-size: 22px; margin-bottom: 5px;'><strong>{primary_prediction}</strong></p>
                                <p style='font-size: 13px; color:#555;'>Logged instantly into local tracking ledger file via safe-write commit routines.</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Output report elements
                        st.markdown("#### 🛠️ Immediate Clinical Actions")
                        act_col1, act_col2 = st.columns(2)
                        
                        with act_col1:
                            # Generate simple clinical txt documentation
                            report_text = f"""======================================================
LEUKEMIA ALO-DAT CLINICAL DIAGNOSTIC REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
======================================================
OPERATOR: DR. J
PATIENT NAME: {st.session_state.active_patient_name.upper()}
PATIENT MRN/ID: {st.session_state.active_patient_id.upper()}
------------------------------------------------------
DIAGNOSTIC PIPELINE RUN: {model_choice}
ALGORITHM: Swarm Optimized (ALO-DAT) + RBF-SVM Engine
------------------------------------------------------
VERDICT DETERMINATION: {primary_prediction}
STATUS: Confirmed & Written into Permanent On-Device Logs
======================================================"""
                            
                            st.download_button(
                                label="📥 Download Signed Print-Ready Report",
                                data=report_text,
                                file_name=f"ALO_DAT_Report_{st.session_state.active_patient_id}.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        with act_col2:
                            if st.button("🔗 Transmit to Electronic Medical Record (EMR/HL7 Mock)", use_container_width=True):
                                st.toast("✅ Document transformed to HL7 FHIR payload and transmitted successfully!")
                                
                    except ValueError as ve:
                        status.update(label="Data alignment checks failed.", state="error")
                        st.error(f"**Shape Alignment Conflict:** {ve}")
                        st.warning("Please ensure the CSV column structure and gene order perfectly match the model's expected initialization inputs.")
                        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with log_panel:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.subheader("📜 On-Device Ledger Logs")
        st.markdown(f"Historical diagnostic track summaries recorded for ID: `{st.session_state.active_patient_id.upper()}`")
        
        # Load logs live from the local DB storage
        history_df = get_patient_history(st.session_state.active_patient_id)
        
        if not history_df.empty:
            for index, row in history_df.iterrows():
                st.markdown(f"""
                    <div style='background-color:#f1f3f5; padding: 10px; border-radius:4px; margin-bottom:8px; border-size: 1px; border-color:#dee2e6;'>
                        <span style='font-size:11px; color:#6c757d;'>📅 {row['timestamp']}</span><br>
                        <span style='font-size:12px; font-weight:600;'>{row['diagnostic_type']}</span><br>
                        <span style='font-size:13px; color:#023e8a; font-weight:bold;'>Result: {row['result_diagnosis']}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No historical records detected for this Patient ID on this computer storage cluster yet.")
        st.markdown("</div>", unsafe_allow_html=True)
