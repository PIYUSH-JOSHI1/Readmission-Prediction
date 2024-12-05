import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_synthetic_readmission_data(n_samples=1000):
    """
    Generate synthetic hospital readmission dataset
    
    Features:
    - Gender: 0 (Female), 1 (Male), 2 (Other)
    - Admission_Type: 0 (Elective), 1 (Emergency), 2 (Urgent)
    - Diagnosis: 0 (Diabetes), 1 (Heart Disease), 2 (Infection), 3 (Injury)
    - Num_Lab_Procedures: 1-100
    - Num_Medications: 1-35
    - Num_Outpatient_Visits: 0-4
    - Num_Inpatient_Visits: 0-4
    - Num_Emergency_Visits: 0-4
    - Num_Diagnoses: 1-9
    - A1C_Result: 0 (Abnormal), 1 (Normal)
    - Readmission: 0 (No), 1 (Yes)
    """
    np.random.seed(42)
    
    # Create synthetic data with some realistic correlations
    data = {
        'Gender': np.random.choice([0, 1, 2], n_samples),
        'Admission_Type': np.random.choice([0, 1, 2], n_samples),
        'Diagnosis': np.random.choice([0, 1, 2, 3], n_samples),
        'Num_Lab_Procedures': np.random.randint(1, 100, n_samples),
        'Num_Medications': np.random.randint(1, 36, n_samples),
        'Num_Outpatient_Visits': np.random.randint(0, 5, n_samples),
        'Num_Inpatient_Visits': np.random.randint(0, 5, n_samples),
        'Num_Emergency_Visits': np.random.randint(0, 5, n_samples),
        'Num_Diagnoses': np.random.randint(1, 10, n_samples),
        'A1C_Result': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate readmission with some realistic probabilities
    # Higher probability of readmission for certain conditions
    readmission_probs = {
        'Diabetes': 0.3,
        'Heart Disease': 0.4,
        'Infection': 0.2,
        'Injury': 0.1
    }
    
    diagnosis_map = {0: 'Diabetes', 1: 'Heart Disease', 2: 'Infection', 3: 'Injury'}
    df['Readmission'] = df['Diagnosis'].map(lambda x: np.random.choice([0, 1], p=[1-readmission_probs[diagnosis_map[x]], readmission_probs[diagnosis_map[x]]]))
    
    return df

def main():
    # Ensure data folder exists
    import os
    os.makedirs('data', exist_ok=True)
    
    # Generate full dataset
    df = generate_synthetic_readmission_data(n_samples=1000)
    
    # Split into train and test sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Save datasets
    train_df.to_csv('data/train_data.csv', index=False)
    test_df.to_csv('data/test_data.csv', index=False)
    
    # Print some basic information
    print("Dataset Generation Complete:")
    print(f"Total Samples: {len(df)}")
    print(f"Training Samples: {len(train_df)}")
    print(f"Test Samples: {len(test_df)}")
    print("\nReadmission Distribution:")
    print(df['Readmission'].value_counts(normalize=True))
    print("\nDiagnosis Distribution:")
    print(df['Diagnosis'].map({0: 'Diabetes', 1: 'Heart Disease', 2: 'Infection', 3: 'Injury'}).value_counts(normalize=True))

if __name__ == "__main__":
    main()