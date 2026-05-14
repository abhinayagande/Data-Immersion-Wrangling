
# ==========================================
# FINAL DATA IMMERSION & WRANGLING SCRIPT
# ==========================================

import pandas as pd
import numpy as np
import os
from datetime import datetime

print("=========================================")
print("DATA CLEANING PROCESS STARTED")
print("=========================================\n")

# ------------------------------------------
# 1. CHECK CURRENT DIRECTORY
# ------------------------------------------

print("📂 Current Directory:", os.getcwd())
print("📄 Files in Directory:", os.listdir())
print("\n-----------------------------------------\n")

# ------------------------------------------
# 2. AUTO-DETECT CSV FILE
# ------------------------------------------

csv_files = [file for file in os.listdir() if file.endswith(".csv")]

if not csv_files:
    print("❌ No CSV file found in this folder.")
    print("👉 Please place your dataset (.csv file) in this folder.")
    exit()

file_name = csv_files[0]
print("✅ CSV File Found:", file_name)

# ------------------------------------------
# 3. LOAD DATASET (WITH SAFE ENCODING)
# ------------------------------------------

try:
    df = pd.read_csv(file_name, encoding="latin1")
    print("✅ Dataset Loaded Successfully!\n")
except Exception as e:
    print("❌ Error while loading file:", e)
    exit()

# ------------------------------------------
# 4. INITIAL EXPLORATION
# ------------------------------------------

print("========== INITIAL DATA INFO ==========")
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nData Types:\n", df.dtypes)
print("\nFirst 5 Rows:\n", df.head())
print("\n---------------------------------------\n")

# ------------------------------------------
# 5. DATA PROFILING
# ------------------------------------------

print("========== DATA PROFILING ==========")
print("\nMissing Values:\n", df.isnull().sum())
print("\nDuplicate Rows:", df.duplicated().sum())
print("\n------------------------------------\n")

# ------------------------------------------
# 6. DATA CLEANING
# ------------------------------------------

print("========== CLEANING DATA ==========")

# Remove duplicates
df.drop_duplicates(inplace=True)

# Standardize column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Convert Amount column if exists
if "Amount" in df.columns:
    df["Amount"] = pd.to_numeric(df["Amount"], errors='coerce')
    df["Amount"].fillna(df["Amount"].median(), inplace=True)
    df = df[df["Amount"] > 0]

# Convert Date columns if exist
date_columns = ["Transaction_Date", "Date_of_Birth"]

for col in date_columns:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Fill missing text values
text_cols = df.select_dtypes(include="object").columns

for col in text_cols:
    df[col] = df[col].fillna("Unknown")
    df[col] = df[col].astype(str).str.strip().str.title()

print("✅ Cleaning Completed\n")

# ------------------------------------------
# 7. FEATURE ENGINEERING
# ------------------------------------------

print("========== FEATURE ENGINEERING ==========")

# Customer Age
if "Date_of_Birth" in df.columns:
    current_year = datetime.now().year
    df["Customer_Age"] = current_year - df["Date_of_Birth"].dt.year

# Transaction Year & Month
if "Transaction_Date" in df.columns:
    df["Transaction_Year"] = df["Transaction_Date"].dt.year
    df["Transaction_Month"] = df["Transaction_Date"].dt.month

print("✅ Feature Engineering Completed\n")

# ------------------------------------------
# 8. OUTLIER HANDLING (IQR METHOD)
# ------------------------------------------

if "Amount" in df.columns:
    Q1 = df["Amount"].quantile(0.25)
    Q3 = df["Amount"].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    df = df[(df["Amount"] >= lower_bound) & (df["Amount"] <= upper_bound)]

print("✅ Outlier Handling Completed\n")

# ------------------------------------------
# 9. FINAL VALIDATION
# ------------------------------------------

print("========== FINAL DATA INFO ==========")
print("Final Shape:", df.shape)
print("\nRemaining Missing Values:\n", df.isnull().sum())

# ------------------------------------------
# 10. SAVE CLEANED FILE
# ------------------------------------------

output_file = "cleaned_dataset.csv"
df.to_csv(output_file, index=False)

print("\n=========================================")
print("🎉 DATA CLEANING SUCCESSFUL!")
print("📁 Cleaned file saved as:", output_file)
print("=========================================")