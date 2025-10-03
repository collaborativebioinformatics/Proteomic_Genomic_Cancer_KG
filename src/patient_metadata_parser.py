import pandas as pd
from loguru import logger
from sklearn.preprocessing import StandardScaler
import numpy as np

RELEVANT_COLS = [
    "TCGA participant ID",
    "MSI.status",
    "Proteomic.subtype",
    "Hypermutated",
    "POLE mutation",
    "BRAF mutation",
    "KRAS mutation",
    "TP53 mutation",
    "18q loss",
    "Tumor.stage",
    "Gender",
    "age_at_initial_pathologic_diagnosis",
    "Tumor.site",
    "Methylation.subtype",
    "Transcriptomic.subtype",
    "vital_status",
    "MLH1_silencing",
    "nonsilent_mutrate",
]


def prepare_s1():
    # Metadata for 95 patients
    df = pd.read_excel(
        "data/input/NIHMS645002-supplement-Supplementary_Tables.xlsx",
        sheet_name="S1_sample_annotation",
        skiprows=56,
    )
    patient_metadata = df[RELEVANT_COLS].copy()
    # Make them graph-safe (no special characters)
    patient_metadata["patient_id"] = patient_metadata[
        "TCGA participant ID"
    ].str.replace("-", "_")

    # Convert MSI status to binary (1 for MSI-H OR MSI-L, 0 for MSS)
    patient_metadata["MSI_binary"] = (
        patient_metadata["MSI.status"].isin(["MSI-H", "MSI-L"])
    ).astype(int)

    patient_metadata["is_subtype_A"] = (
        patient_metadata["Proteomic.subtype"] == "A"
    ).astype(int)
    patient_metadata["is_subtype_B"] = (
        patient_metadata["Proteomic.subtype"] == "B"
    ).astype(int)
    patient_metadata["is_subtype_C"] = (
        patient_metadata["Proteomic.subtype"] == "C"
    ).astype(int)
    patient_metadata["is_subtype_D"] = (
        patient_metadata["Proteomic.subtype"] == "D"
    ).astype(int)
    patient_metadata["is_subtype_E"] = (
        patient_metadata["Proteomic.subtype"] == "E"
    ).astype(int)

    patient_metadata["is_hypermutated"] = (
        patient_metadata["Hypermutated"] == "Hyp"
    ).astype(int)

    # Gender
    patient_metadata["gender_encoded"] = (patient_metadata["Gender"] == "Male").astype(
        int
    )

    # Tumor site (left = 0, right = 1)
    patient_metadata["site_right"] = (
        patient_metadata["Tumor.site"]
        .str.contains("right", case=False, na=False)
        .astype(int)
    )

    patient_metadata["stage_numeric"] = patient_metadata["Tumor.stage"].apply(
        parse_stage
    )

    # === Handling missing values ===
    # Fill missing mutations with 0 (assume wild-type if not reported)
    mutation_cols = [
        "POLE mutation",
        "BRAF mutation",
        "KRAS mutation",
        "TP53 mutation",
        "18q loss",
    ]
    patient_metadata[mutation_cols] = patient_metadata[mutation_cols].fillna(0.0)

    # Missing age = median
    patient_metadata["age_at_initial_pathologic_diagnosis"].fillna(
        patient_metadata["age_at_initial_pathologic_diagnosis"].median(),
        inplace=True,
    )

    patient_metadata["age_at_initial_pathologic_diagnosis"] = patient_metadata[
        "age_at_initial_pathologic_diagnosis"
    ].astype(int)

    # === STEP 6: Create feature matrix for patient nodes ===
    # These will be the initial node features for GNN
    patient_features = patient_metadata[
        [
            "age_at_initial_pathologic_diagnosis",
            "stage_numeric",
            "gender_encoded",
            "site_right",
            "is_hypermutated",
            "POLE mutation",
            "BRAF mutation",
            "KRAS mutation",
            "TP53 mutation",
            "18q loss",
            "nonsilent_mutrate",
        ]
    ].fillna(0)

    # Normalize features (important for GNN!)
    scaler = StandardScaler()
    patient_features_normalised = scaler.fit_transform(patient_features)

    patient_metadata.to_csv("data/output/patient_metadata_processed.csv", index=False)
    np.save("data/output/patient_features.npy", patient_features_normalised)

    logger.info(f"Processed {len(patient_metadata)} patients")
    logger.info(f"\nMSI (H+L): {patient_metadata['MSI_binary'].sum()}")
    logger.info(f"MSS: {(1 - patient_metadata['MSI_binary']).sum()}")
    logger.info("\nProteomic subtypes:")
    for subtype in ["A", "B", "C", "D", "E"]:
        count = (patient_metadata["Proteomic.subtype"] == subtype).sum()
        logger.info(f"  Subtype {subtype}: {count}")


def parse_stage(stage_str):
    if pd.isna(stage_str):
        return np.nan
    stage_str = str(stage_str).upper()
    if "IV" in stage_str:
        return 4
    elif "III" in stage_str:
        return 3
    elif "II" in stage_str:
        return 2
    elif "I" in stage_str:
        return 1
    return np.nan

if __name__ == "__main__":
    prepare_s1()
