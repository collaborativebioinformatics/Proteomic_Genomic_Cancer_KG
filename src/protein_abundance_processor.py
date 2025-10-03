import pandas as pd
import numpy as np
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

# Key proteins from Zhang et al. paper that we want to ensure are included
KEY_PROTEINS = {
    'MMR_proteins': ['MLH1', 'MSH2', 'MSH6', 'PMS2'],  # DNA Mismatch Repair
    'EMT_markers': ['CDH1', 'VIM', 'COL1A1', 'COL3A1', 'FN1'],  # Epithelial-Mesenchymal Transition
    'cell_adhesion': ['CTNNB1', 'CTNNA1', 'DSG2', 'PKP2'],  # Cell Adhesion
    'proliferation': ['MKI67', 'PCNA'],  # Cell Proliferation
    'immune': ['CD3E', 'CD8A']  # Immune markers
}

def load_and_align_protein_data():
    """
    Load S4 protein abundance data and align with S1 metadata.
    Returns aligned protein matrix and metadata.
    """
    logger.info("Loading protein abundance data from S4...")
    
    # Load S4 protein data
    protein_df = pd.read_excel(
        'data/input/NIHMS645002-supplement-Supplementary_Tables.xlsx',
        sheet_name='S4_sample95_count_quantilelog',
        index_col=0
    )
    
    # Load S1 metadata  
    metadata_df = pd.read_csv('data/output/patient_metadata_processed.csv')
    
    logger.info(f"Original protein matrix: {protein_df.shape}")
    logger.info(f"Metadata patients: {len(metadata_df)}")
    
    # Create mapping from S4 sample IDs to base participant IDs
    sample_to_participant = {}
    participant_to_sample = {}
    
    for col in protein_df.columns:
        base_id = '-'.join(col.split('-')[:3])  # Extract TCGA-XX-XXXX
        sample_to_participant[col] = base_id
        participant_to_sample[base_id] = col
    
    # Find patients present in both datasets
    s1_patients = set(metadata_df['TCGA participant ID'])
    s4_patients = set(sample_to_participant.values())
    overlap_patients = s1_patients.intersection(s4_patients)
    
    logger.info(f"Patients in both S1 and S4: {len(overlap_patients)}")
    
    # Filter metadata to overlapping patients
    aligned_metadata = metadata_df[
        metadata_df['TCGA participant ID'].isin(overlap_patients)
    ].copy()
    
    # Filter protein data to overlapping patients and reorder
    overlapping_samples = [participant_to_sample[pid] for pid in overlap_patients 
                          if pid in participant_to_sample]
    aligned_protein_df = protein_df[overlapping_samples].copy()
    
    # Rename protein matrix columns to participant IDs for consistency
    aligned_protein_df.columns = [sample_to_participant[col] for col in aligned_protein_df.columns]
    
    # Ensure both datasets have same patient order
    patient_order = aligned_metadata['TCGA participant ID'].tolist()
    aligned_protein_df = aligned_protein_df[patient_order]
    aligned_metadata = aligned_metadata.set_index('TCGA participant ID').loc[patient_order].reset_index()
    
    logger.info(f"Final aligned dataset: {aligned_protein_df.shape[0]} proteins, {aligned_protein_df.shape[1]} patients")
    
    return aligned_protein_df, aligned_metadata

def filter_to_relevant_proteins(protein_df, n_variable_proteins=800):
    """
    Filter protein matrix to most relevant proteins for KG:
    1. Key proteins from Zhang paper
    2. Most variable proteins across patients
    """
    logger.info("Filtering to most relevant proteins...")
    
    all_proteins = protein_df.index.tolist()
    selected_proteins = set()
    
    # 1. Add key proteins from Zhang paper
    key_protein_list = []
    for category, proteins in KEY_PROTEINS.items():
        for protein in proteins:
            if protein in all_proteins:
                selected_proteins.add(protein)
                key_protein_list.append(protein)
                logger.info(f"✅ Key protein {protein} ({category}) - FOUND")
            else:
                logger.warning(f"❌ Key protein {protein} ({category}) - NOT FOUND")
    
    logger.info(f"Added {len(key_protein_list)} key proteins")
    
    # 2. Find most variable proteins (excluding key proteins already added)
    remaining_proteins = [p for p in all_proteins if p not in selected_proteins]
    remaining_protein_df = protein_df.loc[remaining_proteins]
    
    # Calculate variance for each protein across patients
    protein_variances = remaining_protein_df.var(axis=1).sort_values(ascending=False)
    
    # Select top variable proteins
    n_remaining_needed = n_variable_proteins - len(selected_proteins)
    top_variable = protein_variances.head(n_remaining_needed).index.tolist()
    
    selected_proteins.update(top_variable)
    logger.info(f"Added {len(top_variable)} most variable proteins")
    
    # Create final filtered matrix
    final_proteins = list(selected_proteins)
    filtered_protein_df = protein_df.loc[final_proteins]
    
    logger.info(f"Final protein matrix: {filtered_protein_df.shape}")
    logger.info(f"Key proteins included: {len(key_protein_list)}")
    logger.info(f"Variable proteins added: {len(top_variable)}")
    
    # Save protein selection info
    protein_info = pd.DataFrame({
        'protein': final_proteins,
        'is_key_protein': [p in key_protein_list for p in final_proteins],
        'variance': [protein_df.loc[p].var() for p in final_proteins]
    })
    
    return filtered_protein_df, protein_info

def prepare_kg_datasets():
    """
    Main function to prepare datasets for knowledge graph construction.
    """
    logger.info("=== PREPARING DATASETS FOR KNOWLEDGE GRAPH ===")
    
    # Step 1: Load and align data
    protein_df, metadata_df = load_and_align_protein_data()
    
    # Step 2: Filter to relevant proteins
    filtered_protein_df, protein_info = filter_to_relevant_proteins(protein_df)
    
    # Step 3: Remove duplicate patients from metadata (keep first occurrence)
    metadata_df_clean = metadata_df.drop_duplicates(subset=['TCGA participant ID'], keep='first')
    logger.info(f"Removed {len(metadata_df) - len(metadata_df_clean)} duplicate patients from metadata")
    
    # Step 4: Final alignment check
    protein_patients = set(filtered_protein_df.columns)
    metadata_patients = set(metadata_df_clean['TCGA participant ID'])
    final_patients = protein_patients.intersection(metadata_patients)
    
    # Filter both datasets to final patient set
    final_metadata = metadata_df_clean[
        metadata_df_clean['TCGA participant ID'].isin(final_patients)
    ].copy()
    final_protein_df = filtered_protein_df[list(final_patients)].copy()
    
    logger.info(f"Final KG dataset: {len(final_patients)} patients, {filtered_protein_df.shape[0]} proteins")
    
    # Step 5: Check target distribution
    msi_h_count = (final_metadata['MSI.status'] == 'MSI-H').sum()
    msi_l_count = (final_metadata['MSI.status'] == 'MSI-L').sum()
    mss_count = (final_metadata['MSI.status'] == 'MSS').sum()
    
    logger.info(f"MSI-H patients: {msi_h_count}")
    logger.info(f"MSI-L patients: {msi_l_count}")
    logger.info(f"MSI (H+L) patients: {msi_h_count + msi_l_count}")
    logger.info(f"MSS patients: {mss_count}")
    
    # MSI-H subtype breakdown
    msi_h_metadata = final_metadata[final_metadata['MSI.status'] == 'MSI-H']
    msi_h_subtypes = msi_h_metadata['Proteomic.subtype'].value_counts()
    logger.info("MSI-H by proteomic subtype:")
    for subtype, count in msi_h_subtypes.items():
        logger.info(f"  Subtype {subtype}: {count}")
    
    # Step 6: Save processed data
    logger.info("Saving processed datasets...")
    
    # Save metadata
    final_metadata.to_csv('data/output/kg_patient_metadata.csv', index=False)
    
    # Save protein abundance matrix
    final_protein_df.to_csv('data/output/kg_protein_abundance.csv')
    
    # Save protein info
    protein_info.to_csv('data/output/kg_protein_info.csv', index=False)
    
    # Save as numpy arrays for ML
    np.save('data/output/kg_protein_matrix.npy', final_protein_df.values)
    np.save('data/output/kg_patient_labels.npy', 
            (final_metadata['MSI.status'].isin(['MSI-H', 'MSI-L'])).astype(int).values)
    
    logger.info("✅ Knowledge graph datasets prepared successfully!")
    
    return final_protein_df, final_metadata, protein_info

if __name__ == "__main__":
    prepare_kg_datasets()