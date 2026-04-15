import pandas as pd
import numpy as np
import os
import glob

from config.config import PATH_MIMIC

# ---------------------------------------------------------------------------
# Locate csv.gz files
# ---------------------------------------------------------------------------

def find_file(name):
    """Find a csv.gz file case-insensitively in current directory tree."""
    pattern_upper = f"**/{name.upper()}.csv.gz"
    pattern_lower = f"**/{name.lower()}.csv.gz"
    for pattern in [pattern_upper, pattern_lower, f"**/{name}.csv.gz"]:
        matches = glob.glob(os.path.join(PATH_MIMIC, pattern), recursive=True)
        if matches:
            return matches[0]
    raise FileNotFoundError(f"Cannot find {name}.csv.gz")

print("Loading tables...")
icustays = pd.read_csv(find_file("ICUSTAYS"), parse_dates=["INTIME", "OUTTIME"])
patients = pd.read_csv(find_file("PATIENTS"), parse_dates=["DOB"])
admissions = pd.read_csv(find_file("ADMISSIONS"), parse_dates=["DEATHTIME"])
diagnoses_icd = pd.read_csv(find_file("DIAGNOSES_ICD"), dtype={"ICD9_CODE": str})
services = pd.read_csv(find_file("SERVICES"))
noteevents = pd.read_csv(find_file("NOTEEVENTS"), parse_dates=["CHARTTIME"])

# Normalise column names to uppercase
for df in [icustays, patients, admissions, diagnoses_icd, services, noteevents]:
    df.columns = df.columns.str.upper()

# ---------------------------------------------------------------------------
# Build cohort
# ---------------------------------------------------------------------------
print("Building cohort...")

# Filter ICU stays with LOS >= 2
icu = icustays[icustays["LOS"] >= 2].copy()

# Rank ICU stays per patient by intime (first stay only)
icu["ICUSTAY_ORDER"] = icu.groupby("SUBJECT_ID")["INTIME"].rank(method="first")
icu = icu[icu["ICUSTAY_ORDER"] == 1].copy()

# Join with patients to compute age
icu = icu.merge(patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID", how="inner")
icu["AGE"] = icu.apply(
    lambda r: (r["INTIME"].to_pydatetime() - r["DOB"].to_pydatetime()).total_seconds()
    / (60 * 60 * 24 * 365.242),
    axis=1,
)

# Filter age > 16
icu = icu[icu["AGE"] > 16].copy()

# Join with admissions for mort_icu
icu = icu.merge(
    admissions[["SUBJECT_ID", "HADM_ID", "DEATHTIME"]],
    on=["SUBJECT_ID", "HADM_ID"],
    how="inner",
)
icu["MORT_ICU"] = (
    (icu["DEATHTIME"] >= icu["INTIME"]) & (icu["DEATHTIME"] <= icu["OUTTIME"])
).astype(int)
# Handle NaT deathtime -> 0
icu["MORT_ICU"] = icu["MORT_ICU"].fillna(0).astype(int)

# Keep one row per subject (mirroring DISTINCT ON(subject_id))
cohort = icu.drop_duplicates(subset="SUBJECT_ID", keep="first")[
    ["SUBJECT_ID", "HADM_ID", "ICUSTAY_ID", "AGE", "MORT_ICU", "INTIME"]
].copy()

print(f"Cohort size: {len(cohort)} patients")

# ---------------------------------------------------------------------------
# Join with icustays for LOS, then noteevents within first 2 days
# ---------------------------------------------------------------------------
print("Joining with noteevents...")

# Get LOS from icustays
cohort = cohort.merge(
    icustays[["HADM_ID", "ICUSTAY_ID", "LOS", "INTIME"]].rename(
        columns={"INTIME": "ICU_INTIME"}
    ),
    on=["HADM_ID", "ICUSTAY_ID"],
    how="inner",
    suffixes=("", "_icu"),
)
# Use ICU_INTIME for time calculations
if "INTIME" in cohort.columns:
    cohort.drop(columns=["INTIME"], inplace=True)

# Join with noteevents on subject_id and hadm_id
notes = noteevents.merge(cohort, on=["SUBJECT_ID", "HADM_ID"], how="inner")

# Filter: charttime within first 2 days of ICU intime and not null
notes = notes[notes["CHARTTIME"].notna()].copy()
notes = notes[
    (notes["CHARTTIME"] >= notes["ICU_INTIME"])
    & (notes["CHARTTIME"] <= notes["ICU_INTIME"] + pd.Timedelta(days=2))
].copy()

# Compute icu_time_hr (rounded to nearest hour, mirroring SQL ROUND(..., 0))
notes["ICU_TIME_HR"] = (
    (notes["CHARTTIME"] - notes["ICU_INTIME"]).dt.total_seconds() / 3600
).round(0)

# Trim category (mirroring SQL TRIM)
notes["CATEGORY"] = notes["CATEGORY"].str.strip()

# ---------------------------------------------------------------------------
# Select final columns and sort
# ---------------------------------------------------------------------------
output_cols = [
    "SUBJECT_ID",
    "HADM_ID",
    "ICUSTAY_ID",
    "AGE",
    "MORT_ICU",
    "LOS",
    "ROW_ID",
    "ISERROR",
    "CATEGORY",
    "DESCRIPTION",
    "TEXT",
    "ICU_TIME_HR",
]
# Keep only columns that exist
output_cols = [c for c in output_cols if c in notes.columns]

result = notes[output_cols].sort_values(
    ["SUBJECT_ID", "ICU_TIME_HR"], ascending=True
).reset_index(drop=True)

# Lowercase column names to match original pickle conventions
result.columns = result.columns.str.lower()

print(f"Final dataset: {result.shape[0]} rows, {result.shape[1]} columns")
print(f"Columns: {list(result.columns)}")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
output_name = "./MIMICIII_dataset_medical_notes.pickle"
result.to_pickle(output_name)
print(f"Saved to {output_name}")
