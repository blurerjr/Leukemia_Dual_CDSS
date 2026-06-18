import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Leukemia Diagnostic System", page_icon="🧬", layout="wide")

# --- CACHING MODEL LOADS FOR PERFORMANCE ---
@st.cache_resource
def load_assets(model_type):
    """Loads the model, scaler, encoder, signature, and metrics based on selected pipeline."""
    # Define folder based on user selection
    folder = "models/binary/" if model_type == "Binary Classification" else "models/multi/"
    
    # Load all assets
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

# --- INFERENCE PIPELINE ---
def process_and_predict(df, model, scaler, le, signature):
    """Replicates the notebook preprocessing and prediction steps exactly."""
    # 1. Extract Sample IDs if they exist
    if 'samples' in df.columns:
        sample_ids = df['samples'].copy()
        df = df.drop(columns=['samples'])
    else:
        sample_ids = pd.Series([f"Patient_{i+1}" for i in range(len(df))])
        
    # Drop target column if it was accidentally included in the test file
    if 'type' in df.columns:
        df = df.drop(columns=['type'])

    # 2. Log2 Transformation
    X_log2 = np.log2(df + 1)

    # 3. Min-Max Scaling (Expected to see the full raw gene count)
    X_scaled = scaler.transform(X_log2)
    X_scaled_df = pd.DataFrame(X_scaled, columns=df.columns)

    # 4. Filter using the ALO-DAT signature
    selected_indices = signature['selected_indices']
    X_final = X_scaled_df.iloc[:, selected_indices].values

    # 5. Prediction
    predictions_encoded = model.predict(X_final)
    
    # 6. Decode Labels
    predictions_decoded = le.inverse_transform(predictions_encoded)
    
    # Compile Results
    results_df = pd.DataFrame({
        'Sample ID': sample_ids,
        'Predicted Diagnosis': predictions_decoded
    })
    
    return results_df

# --- UI LAYOUT ---
st.title("🧬 Leukemia Gene Expression Diagnostic System")
st.markdown("Upload a patient gene expression CSV to predict leukemia subtypes using our optimized ALO-DAT RBF-SVM pipeline.")

# Sidebar Configuration
st.sidebar.header("System Settings")
model_choice = st.sidebar.radio("Select Diagnostic Model", ["Binary Classification", "Multi-Class Classification"])

# Load appropriate assets
try:
    svm_model, minmax_scaler, label_encoder, gene_signature, model_metrics = load_assets(model_choice)
    st.sidebar.success(f"{model_choice} engine loaded successfully!")
except FileNotFoundError:
    st.sidebar.error("Error: Asset files not found. Ensure your .pkl and .json files are placed in 'models/binary/' and 'models/multi/' directories.")
    st.stop()

# Display Model Metrics Dashboard
with st.expander("📊 View Clinical Performance Metrics (Unseen Test Data)", expanded=False):
    st.markdown(f"**Selected Biomarkers:** {model_metrics['number_of_biomarkers']} Genes")
    cols = st.columns(5)
    cols[0].metric("Accuracy", f"{model_metrics['diagnostic_accuracy_pct']}%")
    cols[1].metric("Precision", f"{model_metrics['macro_precision_pct']}%")
    cols[2].metric("Recall (Sens)", f"{model_metrics['macro_recall_sens_pct']}%")
    cols[3].metric("Specificity", f"{model_metrics['macro_specificity_pct']}%")
    cols[4].metric("F1-Score", f"{model_metrics['macro_f1_score_pct']}%")

st.divider()

# Main Prediction Interface
st.subheader("Patient Inference Panel")
uploaded_file = st.file_uploader("Upload Patient Gene Expression Data (.csv)", type=["csv"])

if uploaded_file is not None:
    st.info("File uploaded successfully. Processing gene expressions...")
    
    # Read the data
    input_data = pd.read_csv(uploaded_file)
    
    # Show a preview of the uploaded data
    with st.expander("Preview Uploaded Data"):
        st.dataframe(input_data.head())
    
    # Run Prediction Button
    if st.button("Run Diagnostic Prediction", type="primary"):
        with st.spinner("Applying Log2 normalization, scaling, and ALO-DAT signature slicing..."):
            try:
                # Execute the exact pipeline
                results = process_and_predict(input_data, svm_model, minmax_scaler, label_encoder, gene_signature)
                
                st.success("Diagnostic processing complete!")
                
                # Display Results
                st.subheader("Prediction Results")
                st.dataframe(results, use_container_width=True)
                
                # Allow user to download the results
                csv_export = results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Diagnostic Report",
                    data=csv_export,
                    file_name="leukemia_diagnostic_results.csv",
                    mime="text/csv",
                )
            except ValueError as e:
                st.error(f"Data Mismatch Error: {e}")
                st.warning("Ensure the uploaded CSV contains the exact same number of gene columns (in the same order) as the original training dataset before the ALO-DAT selection.")
