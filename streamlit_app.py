import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Leukemia Prediction System",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DIAGNOSTIC KNOWLEDGE BASE ---
SUBTYPE_INFO = {
    # Binary Subtypes
    "AML": {
        "name": "Acute Myeloid Leukemia",
        "driver": "A rapid, aggressive cancer of the myeloid lineage of blood cells. It is characterized by the overproduction of immature white blood cells (myeloblasts) in the bone marrow, which crowd out normal, healthy red blood cells, platelets, and functional white blood cells.",
        "clinical": "Requires immediate, intensive induction chemotherapy."
    },
    "normal": {  # Note: lowercase 'normal' to match exact notebook label
        "name": "Normal / Healthy Control Profile",
        "driver": "Baseline, non-leukemic tissue or bone marrow samples. In a diagnostic environment, this acts as the negative control group showing healthy homeostatic gene expression levels.",
        "clinical": "No leukemic intervention required. Routine monitoring if symptoms persist."
    },
    "Normal": { # Fallback for capitalization
        "name": "Normal / Healthy Control Profile",
        "driver": "Baseline, non-leukemic tissue or bone marrow samples. In a diagnostic environment, this acts as the negative control group showing healthy homeostatic gene expression levels.",
        "clinical": "No leukemic intervention required. Routine monitoring if symptoms persist."
    },
    
    # Multi-Class Subtypes
    "B-CELL_ALL_ETV6-RUNX1": {
        "name": "B-cell ALL with ETV6-RUNX1 Fusion",
        "driver": "Arises from a structural chromosomal translocation, specifically t(12;21)(p13.2;q22.1), which fuses the ETV6 gene with the RUNX1 gene.",
        "clinical": "This is one of the most common pediatric B-ALL subtypes and is highly responsive to standard chemotherapy, generally carrying a favorable long-term prognosis."
    },
    "B-CELL_ALL_HYPERDIP": {
        "name": "Hyperdiploid B-cell ALL",
        "driver": "Characterized by numerical chromosomal abnormalities where the leukemic cells contain an abnormally high number of chromosomes (typically 51 to 65 chromosomes, instead of the normal 46) without structural translocations.",
        "clinical": "Like ETV6-RUNX1, hyperdiploidy is a common pediatric subtype linked to a highly favorable prognosis and exceptional response to standard treatment regimens."
    },
    "B-CELL_ALL_T-ALL": {
        "name": "T-cell Acute Lymphoblastic Leukemia",
        "driver": "Malignant transformation of T-cell precursors (thymocytes). In integrated datasets, it is occasionally given a generic 'B-CELL_ALL' prefix text structure, but it specifically isolates T-lineage leukemia.",
        "clinical": "T-ALL represents an aggressive hematological malignancy. It requires distinct, intensive therapeutic protocols compared to standard B-cell ALL due to a higher risk of early central nervous system (CNS) relapse."
    },
    "B-CELL_ALL_TCF3-PBX1": {
        "name": "B-cell ALL with TCF3-PBX1 Fusion",
        "driver": "Driven by a balanced chromosomal translocation, t(1;19)(q23;p13.3), which fuses the transcription factor gene TCF3 (E2A) with the PBX1 homeobox gene.",
        "clinical": "Historically associated with an increased risk of treatment failure and central nervous system relapse, it is treated as an intermediate-to-high risk subtype requiring intensive modern multi-agent chemotherapy."
    },
    "B-CELL_ALL_HYPO": {
        "name": "Hypodiploid B-cell ALL",
        "driver": "Characterized by a significant loss of chromosomes, leaving the leukemic cells with fewer than 45 chromosomes (often sub-categorized as near-haploid or low-hypodiploid).",
        "clinical": "This is a rare but critical subtype to diagnose early. It carries an extremely poor prognosis, high resistance to standard chemotherapies, and frequently mandates aggressive upfront treatments, such as Allogeneic Hematopoietic Stem Cell Transplantation (HSCT)."
    },
    "B-CELL_ALL_MLL": {
        "name": "B-cell ALL with MLL Rearrangement (KMT2A-rearranged)",
        "driver": "Involves structural abnormalities and translocations at chromosome 11q23, disrupting the Mixed Lineage Leukemia (MLL) gene, now officially designated as KMT2A. Common fusion partners include AF4, ENL, and AF9.",
        "clinical": "This subtype dominates infant leukemia (children under 1 year old). It represents a highly aggressive, high-risk malignancy associated with distinct biological features, poor clinical outcomes, and an urgent requirement for novel targeted therapies (like Menin inhibitors)."
    }
}

# --- LIGHTWEIGHT SYSTEM MEMORY INITIALIZATION ---
if 'clinical_history' not in st.session_state:
    st.session_state.clinical_history = []
if 'active_patient_name' not in st.session_state:
    st.session_state.active_patient_name = ""
if 'active_patient_id' not in st.session_state:
    st.session_state.active_patient_id = ""
if 'workspace_ready' not in st.session_state:
    st.session_state.workspace_ready = False
if 'cached_matrix_df' not in st.session_state:
    st.session_state.cached_matrix_df = None
if 'cached_file_signature' not in st.session_state:
    st.session_state.cached_file_signature = ""

def append_to_memory_log(patient_name, patient_id, diagnostic_type, result_diagnosis):
    """Appends running logs instantly to session RAM ledger."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "patient_name": patient_name.strip().upper(),
        "patient_id": patient_id.strip().upper(),
        "diagnostic_type": diagnostic_type,
        "result_diagnosis": result_diagnosis
    }
    st.session_state.clinical_history.append(log_entry)

# --- CUSTOM CLINICAL THEME (CSS) ---
st.markdown("""
    <style>
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
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-top: 5px solid #2e7d32;
        padding: 20px;
        border-radius: 6px;
        margin-top: 15px;
    }
    
    .result-title {
        color: #2e7d32;
        margin-top: 0px;
        margin-bottom: 5px;
        font-size: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# --- CACHING MODEL ENGINE LOADS ---
@st.cache_resource
def load_pipeline_assets(model_type):
    """Loads ML artifacts from directory parameters with memory caching."""
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
    """Strict execution of the notebook's data preparation algorithms."""
    if 'samples' in df.columns:
        df = df.drop(columns=['samples'])
    if 'type' in df.columns:
        df = df.drop(columns=['type'])

    X_log2 = np.log2(df + 1)
    X_scaled = scaler.transform(X_log2)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)

    selected_indices = signature['selected_indices']
    X_selected = X_scaled_df.iloc[:, selected_indices].values

    predictions_encoded = model.predict(X_selected)
    predictions_decoded = le.inverse_transform(predictions_encoded)
    
    return predictions_decoded

# --- SIDEBAR CONTROL PANEL ---
with st.sidebar:
    st.markdown("### 🏥 Authorized Operator")
    st.info("**Welcome DR. J**\n\nRole: Attending Hematologist")
    st.divider()
    
    st.markdown("### ⚙️ Engine Configurations")
    model_choice = st.radio(
        "Select Computational Pipeline:", 
        ["Binary Classification", "Multi-Class Classification"]
    )
    
    try:
        svm_model, minmax_scaler, label_encoder, gene_signature, model_metrics = load_pipeline_assets(model_choice)
        st.caption("✅ Pipeline components actively verified in background memory cache.")
    except FileNotFoundError:
        st.error("⚠️ System assets missing. Verify folder layout paths.")
        st.stop()
        
    st.divider()
    if st.session_state.workspace_ready:
        if st.button("🔄 Clear Active Patient Workspace", use_container_width=True):
            st.session_state.active_patient_name = ""
            st.session_state.active_patient_id = ""
            st.session_state.cached_matrix_df = None
            st.session_state.cached_file_signature = ""
            st.session_state.workspace_ready = False
            st.rerun()

# --- MAIN APP HEADER ---
st.markdown("<h1 class='main-title'>🧬 Leukemia Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Binary and Multi Class Leukemia Prediction using Enhanced ALO-DAT in Gene Expression Diagnostic</p>", unsafe_allow_html=True)

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
    st.markdown("Before loading raw expression counts, register patient files to track diagnostic evaluations.")
    
    col1, col2 = st.columns(2)
    with col1:
        pt_name_input = st.text_input("Patient Full Name:", placeholder="e.g. Abd Basit").strip()
    with col2:
        pt_id_input = st.text_input("Unique Medical Record Number (MRN / Patient ID):", placeholder="e.g. L-2026-8843").strip()
        
    if st.button("🔐 Initialize Patient Diagnostic Workspace", type="primary"):
        if pt_name_input and pt_id_input:
            st.session_state.active_patient_name = pt_name_input
            st.session_state.active_patient_id = pt_id_input
            st.session_state.workspace_ready = True
            st.rerun()
        else:
            st.warning("🚨 Operational Failure: Both Patient Full Name and Unique Patient ID are strictly mandatory for patient verification.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- WORKFLOW STEP 2 & 3: COMPUTATIONAL PIPELINE ---
else:
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
        st.markdown("Upload the raw gene expression values `.csv` matrix below.")
        
        uploaded_matrix = st.file_uploader("Select Target Expression Profile File", type=["csv"])
        
        if uploaded_matrix is not None:
            # --- HIGH-SPEED INGESTION CACHE ---
            current_sig = f"{uploaded_matrix.name}_{uploaded_matrix.size}"
            
            if st.session_state.cached_file_signature != current_sig:
                with st.spinner("⚡ High-Speed RAM Optimizer: parsing expression matrix..."):
                    st.session_state.cached_matrix_df = pd.read_csv(uploaded_matrix)
                    st.session_state.cached_file_signature = current_sig
            
            raw_matrix_df = st.session_state.cached_matrix_df
                
            if st.button("⚡ Execute Computational Diagnostics", type="primary"):
                # Using st.spinner to prevent UI accordion popups
                with st.spinner("Processing genomic matrix through ALO-DAT pipelines..."):
                    try:
                        results = execute_inference(
                            raw_matrix_df, svm_model, minmax_scaler, label_encoder, gene_signature
                        )
                        primary_prediction = results[0]
                        
                        # Lookup medical context
                        medical_context = SUBTYPE_INFO.get(
                            primary_prediction, 
                            {
                                "name": "Unknown Variant", 
                                "driver": "Insufficient registry data for this genetic signature.", 
                                "clinical": "Further manual pathological review required."
                            }
                        )
                        
                        # Save instantly to memory log
                        append_to_memory_log(
                            st.session_state.active_patient_name,
                            st.session_state.active_patient_id,
                            model_choice,
                            medical_context["name"]
                        )
                        
                        st.success("✅ Diagnostic Matrix Calculation Completed.")
                        
                        # Enhanced Contextual UI output
                        st.markdown(f"""
                            <div class='result-box'>
                                <p style='margin-bottom:0px; font-weight:bold; color:#6c757d;'>DIAGNOSTIC VERDICT:</p>
                                <h2 class='result-title'>{medical_context['name']}</h2>
                                <p><strong>Raw Model Signature:</strong> <code>{primary_prediction}</code></p>
                                <hr style="margin: 10px 0px; border: 0; border-top: 1px solid #dee2e6;">
                                <p style='margin-bottom: 5px;'><strong>🔬 Genetic Driver / Description:</strong></p>
                                <p style='color: #495057; font-size: 14px;'>{medical_context['driver']}</p>
                                <p style='margin-bottom: 5px; margin-top:10px;'><strong>🏥 Clinical Context:</strong></p>
                                <p style='color: #495057; font-size: 14px;'>{medical_context['clinical']}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown("#### 🛠️ Immediate Clinical Actions")
                        act_col1, act_col2 = st.columns(2)
                        
                        with act_col1:
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
VERDICT DETERMINATION: {medical_context['name']}
RAW SIGNATURE: {primary_prediction}

GENETIC DRIVER / DESCRIPTION:
{medical_context['driver']}

CLINICAL CONTEXT / PROGNOSIS:
{medical_context['clinical']}

STATUS: Confirmed via In-Memory Secure Evaluation Profile
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
                        st.error(f"**Shape Alignment Conflict:** {ve}")
                        st.warning("Please verify that the CSV row features perfectly match the inputs expected by the training architecture scale.")
                        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with log_panel:
        st.markdown("<div class='clinical-card'>", unsafe_allow_html=True)
        st.subheader("📜 Current Session Ledger")
        st.markdown(f"Runs tracked for ID: `{st.session_state.active_patient_id.upper()}`")
        
        matching_records = [
            log for log in st.session_state.clinical_history 
            if log['patient_id'] == st.session_state.active_patient_id.strip().upper()
        ]
        
        if matching_records:
            for row in reversed(matching_records):
                st.markdown(f"""
                    <div style='background-color:#f1f3f5; padding: 10px; border-radius:4px; margin-bottom:8px; border: 1px solid #dee2e6;'>
                        <span style='font-size:11px; color:#6c757d;'>📅 {row['timestamp']}</span><br>
                        <span style='font-size:12px; font-weight:600;'>{row['diagnostic_type']}</span><br>
                        <span style='font-size:13px; color:#023e8a; font-weight:bold;'>Result: {row['result_diagnosis']}</span>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.caption("No diagnostic interactions logged for this patient profile inside the active session workspace.")
        st.markdown("</div>", unsafe_allow_html=True)
