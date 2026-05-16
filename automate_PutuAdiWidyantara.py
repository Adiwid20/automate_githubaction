import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from pathlib import Path

def run_preprocessing():
    print("Memulai proses otomatisasi preprocessing...")

        # 1. Setup Path Dinamis (Menyesuaikan di mana script ini dijalankan)
    
    PATH = "dataset"
    raw_data_path = os.path.join(PATH, 'loan_data.csv')
    clean_data_path = os.path.join(PATH, 'loan_data_pre.csv')
    
    # 2. Load Data Mentah
    try:
        df = pd.read_csv(raw_data_path)
        print(f"Data mentah berhasil dimuat dengan shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File {raw_data_path} tidak ditemukan.")
        return

    # 3. Handle Null Value
    df.dropna(inplace=True)
    
    # 4. Handle Outlier (Clipping dengan IQR)
    df_pre = df.copy()
    num_features = df.select_dtypes(include=[np.number]).columns.drop('loan_status')
    
    for col in num_features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_pre[col] = df_pre[col].clip(lower=lower_bound, upper=upper_bound)

    # 5. Binning
    bins_usia = [0, 25, 35, 50, 100]
    labels_age = ['Remaja', 'Dewasa', 'Paruh Baya', 'Lansia']
    df_pre['age_group'] = pd.cut(df_pre['person_age'], bins=bins_usia, labels=labels_age)

    bins_credit = [0, 579, 669, 739, 850]
    labels_credit = ['Poor', 'Fair', 'Good', 'Excellent']
    df_pre['credit_score_group'] = pd.cut(df_pre['credit_score'], bins=bins_credit, labels=labels_credit)

    bins_exp = [-1, 2, 5, 10, 50] 
    labels_exp = ['Fresher', 'Mid-Level', 'Senior', 'Veteran']
    df_pre['emp_exp_group'] = pd.cut(df_pre['person_emp_exp'], bins=bins_exp, labels=labels_exp)

    # 6. Encoding
    df_encode = df_pre.copy()

    # - Binary Encoding
    df_encode['person_gender'] = LabelEncoder().fit_transform(df_encode['person_gender'])
    df_encode['previous_loan_defaults_on_file'] = df_encode['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})

    # - Ordinal Encoding
    education_order = ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']
    age_order = ['Remaja', 'Dewasa', 'Paruh Baya', 'Lansia']
    credit_order = ['Poor', 'Fair', 'Good', 'Excellent']
    exp_order = ['Fresher', 'Mid-Level', 'Senior', 'Veteran']

    oe = OrdinalEncoder(categories=[education_order, age_order, credit_order, exp_order])
    ordinalcolumn = ['person_education', 'age_group', 'credit_score_group', 'emp_exp_group']
    df_encode[ordinalcolumn] = oe.fit_transform(df_encode[ordinalcolumn])

    # - One Hot Encoding
    df_encode = pd.get_dummies(
        df_encode, columns=['person_home_ownership', 'loan_intent'], 
        drop_first=True, dtype=int
    )

    # 7. Pembersihan Kolom Redundan
    kolom_dibuang = ['person_age', 'credit_score', 'person_emp_exp']
    df_encode = df_encode.drop(columns=kolom_dibuang)

    # 8. Simpan Dataset Hasil Preprocessing
    df_encode.to_csv(clean_data_path, index=False)
    print(f"Preprocessing selesai! Shape akhir: {df_encode.shape}")
    print(f"Data tersimpan di: {clean_data_path}")

if __name__ == "__main__":
    run_preprocessing()