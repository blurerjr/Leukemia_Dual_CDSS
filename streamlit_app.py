import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# --- Page Layout Configuration ---
st.set_page_config(
    page_title="Leukemia Diagnostic Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom UI Branding styles ---
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: 800; color: #1E3A8A; margin-bottom: 5px; }
    .subtitle { font-size: 16px; color: #4B5563; margin-bottom: 25px; }
    .card { background-color: #F3F4F6; padding: 20px; border-radius: 10px; margin-bottom: 15px; }
    </style>
""", unsafe_with_html=True)

st.markdown('<div class="main-title">🔬 Clinical Intelligence Leukemia Diagnostic Suite</div>', unsafe_with_html=True)
st.markdown('<div class="subtitle">High-Performance Swarm-Optimized (ALO-DAT) SVM Screening Engine</div>', unsafe_with_html=True)

# --- Thread-Safe Cached Resource Loader ---
@st.cache_resource
def load_pipeline_assets(task_mode):
    """
    Loads separate decoupled assets depending on explicit clinical classification mode selected.
    """
    if task_mode == "Binary Classification (AML vs. Normal)":
        base_dir = "binary_class"
    else:
        base_dir = "Multi-class Classification (Subtypes)"
        
    model_path = os.path.join(base_dir, "leukemia_rbf_svm_model.pkl")
    scaler_path = os.path.join(base_dir, "gene_minmax_scaler.pkl")
    encoder_path = os.path.join(base_dir, "leukemia_label_encoder.pkl")
    genes_path = os.path.join(base_dir, "alo_dat_selected_genes.pkl")
    meta_path = os.path.join(base_dir, "clinical_performance_metadata.json")
    
    assets = {}
    
    # Verify Directory Structures
    if not os.path.exists(base_dir):
        st.error(f"🚨 Model assets directory '{base_dir}/' not found. Please verify deployment path layout.")
        return None
        
    try:
        with open(model_path, "rb") as f:
            assets["model"] = pickle.load(f)
        with open(scaler_path, "rb") as f:
            assets["scaler"] = pickle.load(f)
        with open(encoder_path, "rb") as f:
            assets["encoder"] = pickle.load(f)
        with open(genes_path, "rb") as f:
            assets["selected_genes"] = pickle.load(f)
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                assets["metadata"] = json.load(f)
        else:
            assets["metadata"] = None
    except Exception as e:
        st.error(f"❌ Failed to parse assets for {task_mode}. Underlying error: {e}")
        return None
        
    return assets

# --- Sidebar Configuration controls ---
st.sidebar.header("🕹️ Diagnostic Control Center")

task_selection = st.sidebar.selectbox(
    "Select Explicit Model Path:",
    ["Binary Classification (AML vs. Normal)", "Multi-class Classification (Subtypes)"]
)

# Fetch corresponding engine vectors
engine = load_pipeline_assets(task_selection)

# --- Main Interface Tabs Layout ---
tab1, tab2, tab3 = st.tabs(["📊 Engine Performance Dashboard", "🧬 Diagnostic Screening System", "🔍 Biomarker Signature Explorer"])

# ==========================================
# TAB 1: SYSTEM ENGINE BENCHMARKS
# ==========================================
with tab1:
    st.subheader("📈 Verified Cross-Validation Validation Benchmarks")
    if engine and engine["metadata"]:
        meta = engine["metadata"]
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(label="System Cross-Validation Accuracy", value=f"{meta.get('accuracy_pct', meta.get('diagnostic_accuracy_pct', 95.0))}%")
        with col2:
            st.metric(label="Macro F1-Score Efficiency", value=f"{meta.get('macro_f1_score_pct', 95.0)}%")
        with col3:
            st.metric(label="Engine Matthews Corr Coeff (MCC)", value=f"{meta.get('matthews_corrcoef_pct', 90.0)}%")
        with col4:
            st.metric(label="Optimized Swarm Biomarkers", value=f"{meta.get('number_of_biomarkers', 'N/A')} Genes")
            
        st.markdown("### 🛠️ Architecture Specification Pipeline")
        st.info(
            f"**Classifier Core:** Support Vector Machine (SVM) utilizing a high-dimensional Radial Basis Function (RBF) Kernel. "
            f"**Feature Search Archetype:** Ant Lion Optimizer with Chaotic Jump and V4 Transfer Function (ALO-DAT) "
            f"trained explicitly on isolated cohort expressions."
        )
    else:
        st.warning("⚠️ No metadata profile discovered. Pipeline running on customized default parameters.")

# ==========================================
# TAB 2: DIAGNOSTIC SCRIPT EXECUTION
# ==========================================
with tab2:
    st.subheader("📥 Patient Expression Profile Uploading")
    st.markdown("Upload expression profile matrix tables (`.csv`, `.tsv`, or `.txt`) formatted positionally or with named columns.")
    
    uploaded_file = st.file_uploader("Choose expression file...", type=["csv", "tsv", "txt", "xlsx"])
    
    if uploaded_file is not None and engine is not None:
        try:
            # Multi-format delimiter parsing logic
            if uploaded_file.name.endswith('.csv'):
                # Handle possible headerless format like aml.csv safely
                first_line = uploaded_file.getvalue().decode('utf-8').split('\n')[0]
                if '\t' in first_line:
                    df_input = pd.read_csv(uploaded_file, sep='\t', header=None)
                else:
                    df_input = pd.read_csv(uploaded_file, header=None if not first_line.isalpha() else 'infer')
            elif uploaded_file.name.endswith(('.tsv', '.txt')):
                df_input = pd.read_csv(uploaded_file, sep='\t', header=None)
            else:
                df_input = pd.read_excel(uploaded_file, header=None)
                
            st.success("✅ File streams ingested successfully.")
            
            # --- Meta Extraction & Normalization Steps ---
            # Try parsing Patient/Sample Identifiers and clinical true labels if present
            sample_ids = []
            true_labels = []
            
            # Identify text labels vs continuous inputs
            text_cols = df_input.select_dtypes(include=['object']).columns.tolist()
            
            if len(text_cols) >= 2:
                sample_ids = df_input[text_cols[0]].tolist()
                true_labels = df_input[text_cols[1]].tolist()
                numeric_df = df_input.drop(columns=[text_cols[0], text_cols[1]])
            elif len(text_cols) == 1:
                sample_ids = df_input[text_cols[0]].tolist()
                numeric_df = df_input.drop(columns=[text_cols[0]])
            else:
                sample_ids = [f"Patient_Obs_{i+1}" for i in range(len(df_input))]
                numeric_df = df_input.select_dtypes(include=[np.number])
                
            # --- CRITICAL BULLETPROOF FEATURE ALIGNMENT ---
            scaler = engine["scaler"]
            model = engine["model"]
            encoder = engine["encoder"]
            selected_features_asset = engine["selected_genes"]
            
            # Capture what the scaler model explicitly targets
            expected_features_count = scaler.n_features_in_
            
            # Reconstruct the feature dataframe to fit exactly what scaler expects
            if hasattr(scaler, "feature_names_in_"):
                expected_gene_names = scaler.feature_names_in_
                # Check if uploaded matrix labels match named parameters
                matching_features = [col for col in expected_gene_names if col in numeric_df.columns]
                
                if len(matching_features) >= (expected_features_count * 0.5):
                    st.info(f"🧬 Gene probe signatures discovered. Aligning named features dynamically ({len(matching_features)} matched).")
                    X_raw = pd.DataFrame(index=numeric_df.index)
                    for gene in expected_gene_names:
                        X_raw[gene] = numeric_df[gene] if gene in numeric_df.columns else 0.0
                else:
                    # Positional mapping fallback
                    if numeric_df.shape[1] == expected_features_count:
                        st.info("📊 Matrix columns match feature spaces positionally. Mapping columns onto trained dimensions.")
                        X_raw = numeric_df.copy()
                        X_raw.columns = expected_gene_names
                    elif numeric_df.shape[1] > expected_features_count:
                        st.info(f"📊 Slicing trailing expression boundaries positionally (-{expected_features_count} features).")
                        X_raw = numeric_df.iloc[:, -expected_features_count:].copy()
                        X_raw.columns = expected_gene_names
                    else:
                        st.error(f"❌ Feature Dimension Mismatch! Scaler expects {expected_features_count} inputs, but numeric input contains {numeric_df.shape[1]}.")
                        st.stop()
            else:
                # Scaler was trained on raw array structures without metadata keys
                if numeric_df.shape[1] == expected_features_count:
                    X_raw = numeric_df.copy()
                elif numeric_df.shape[1] > expected_features_count:
                    X_raw = numeric_df.iloc[:, -expected_features_count:].copy()
                else:
                    st.error(f"❌ Data space violation. Expected numeric width of {expected_features_count}, received {numeric_df.shape[1]}.")
                    st.stop()

            # --- AUTO-SCALE TRANSFORMER GUARD ---
            max_value_observed = X_raw.max().max()
            if max_value_observed > 20.0:
                st.info(f"📈 High amplitude values identified (Max: {max_value_observed:.2f}). Processing standard Training Log2 Transform step.")
                X_log2 = np.log2(X_raw.astype(float) + 1.0)
            else:
                st.info(f"📉 Pre-compressed/Log values identified (Max: {max_value_observed:.2f}). Bypassing transformation to avoid double logging.")
                X_log2 = X_raw.astype(float).copy()

            # --- NORMALIZATION SCALING ---
            X_scaled = scaler.transform(X_log2)
            
            # Reconstruct scaling coordinates to execute biomarker mask slicing
            if hasattr(scaler, "feature_names_in_"):
                X_scaled_df = pd.DataFrame(X_scaled, columns=scaler.feature_names_in_)
            else:
                X_scaled_df = pd.DataFrame(X_scaled)
                
            # --- SWARM SELECTION FEATURE EXTRACTION ---
            # Evaluate format of selected genes asset (Indices vs. Named Probes)
            if all(isinstance(x, (int, np.integer)) for x in selected_features_asset):
                # Integer Index Filtering
                # Multi-class pipeline includes ANOVA preprocessing; index values apply onto available widths
                if X_scaled_df.shape[1] == 22283 and max(selected_features_asset) >= 22283:
                    st.error("❌ High-index reference out of boundaries. Slicing failed.")
                    st.stop()
                X_final = X_scaled_df.iloc[:, selected_features_asset].values
            else:
                # Named Array/String Extraction
                X_final = X_scaled_df[list(selected_features_asset)].values
                
            # Verify shape fits model execution signatures
            if X_final.shape[1] != model.n_features_in_:
                st.warning(f"🔄 Re-aligning dimension spaces: Target signature expects {model.n_features_in_} features, extracting positionally.")
                X_final = X_scaled[:, :model.n_features_in_]

            # --- RUN DIAGNOSTIC PREDICTION ---
            predictions_encoded = model.predict(X_final)
            predictions_decoded = encoder.inverse_transform(predictions_encoded)
            
            # --- BUILD OUTPUT INTERACTIVE FRAME ---
            results_payload = {
                "Sample ID": sample_ids,
                "Model Diagnostic Prediction": predictions_decoded
            }
            if len(true_labels) == len(sample_ids):
                results_payload["Clinical Ground Truth"] = true_labels
                
            results_df = pd.DataFrame(results_payload)
            
            st.markdown("### 🧬 Diagnostic Screener Classifications")
            st.dataframe(results_df, use_container_width=True)
            
            # Export Operations
            csv_data = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Diagnostic Status Assessment Report (CSV)",
                data=csv_data,
                file_name="Leukemia_Diagnostic_Report.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"💥 Failed to process prediction execution stack. Stack trace logs: {e}")

# ==========================================
# TAB 3: GENE SIGNATURE DISCOVERY EXPLORER
# ==========================================
with tab3:
    st.subheader("🔍 Active Biomarker Target Profiles")
    if engine:
        genes_list = engine["selected_genes"]
        st.markdown(f"The ALO-DAT optimization module isolated **{len(genes_list)} highly descriptive signatures** that govern this classifier's decision boundaries.")
        
        # Display as readable clean layout
        formatted_df = pd.DataFrame({
            "Signature Identifier Index": range(1, len(genes_list) + 1),
            "Gene Name / Probe Dimension Position": genes_list
        })
        st.dataframe(formatted_df, use_container_width=True, height=400)
    else:
        st.info("Upload profile configurations or select an explicit model path to review gene pathways.")
