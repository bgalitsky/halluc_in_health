"""
Disease Prediction from Symptoms using Machine Learning
This script trains a classification model to predict diseases based on symptoms.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
import kagglehub
# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -------------------------------
# 1. Load and Inspect Data
# -------------------------------
def load_data():


    # Download latest version
    path = kagglehub.dataset_download("uom190346a/disease-symptoms-and-patient-profile-dataset")

    print("Path to dataset files:", path)
    df = pd.read_csv('data/Disease_symptom_and_patient_profile_dataset.csv')  # Update path as needed
    print("Data shape:", df.shape)
    print("\nFirst few rows:")
    print(df.head())
    return df

# -------------------------------
# 2. Exploratory Data Analysis
# -------------------------------
def cat_summary(dataframe, col_name, plot=False):
    """
    Print value counts and ratio for categorical columns.
    Optionally display count plots.
    """
    summary = pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio (%)": 100 * dataframe[col_name].value_counts() / len(dataframe)
    })
    print(summary)
    print("-" * 40)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(8, 6))
        sns.countplot(x=dataframe[col_name], ax=axs[0])
        axs[0].set_title(f"Frequency of {col_name}")
        axs[0].tick_params(axis='x', rotation=90)

        # Pie chart
        values = dataframe[col_name].value_counts()
        axs[1].pie(values, labels=values.index, autopct='%1.1f%%', startangle=90)
        axs[1].set_title(f"Distribution of {col_name}")
        plt.tight_layout()
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    """
    Display descriptive statistics for numerical columns.
    Optionally plot histograms and boxplots.
    """
    print(f"\n--- Summary for {numerical_col} ---")
    print(dataframe[numerical_col].describe().T)

    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(10, 6))
        sns.histplot(dataframe[numerical_col], kde=True, ax=axs[0])
        axs[0].set_title(f"{numerical_col} Distribution")

        sns.boxplot(x=dataframe[numerical_col], ax=axs[1])
        axs[1].set_title(f"{numerical_col} Box Plot")
        plt.tight_layout()
        plt.show()

# -------------------------------
# 3. Feature Engineering
# -------------------------------
def create_features(df):
    """
    Create derived features such as combinations of symptoms,
    disease frequency, risk score, age groups, etc.
    """
    # Copy dataframe
    dfd = df.copy()

    # Combine symptoms
    dfd['Fever_and_Cough'] = (dfd['Fever'] == 'Yes') & (dfd['Cough'] == 'Yes')
    dfd['Fever_and_Fatigue'] = (dfd['Fever'] == 'Yes') & (dfd['Fatigue'] == 'Yes')
    dfd['Fatigue_and_Cough'] = (dfd['Fatigue'] == 'Yes') & (dfd['Cough'] == 'Yes')
    dfd['Fever_and_Fatigue_and_Cough'] = (dfd['Fever'] == 'Yes') & (dfd['Cough'] == 'Yes') & (dfd['Fatigue'] == 'Yes')

    # Disease frequency
    disease_freq = dfd['Disease'].value_counts()
    dfd['Disease_Frequency'] = dfd['Disease'].map(disease_freq)

    # Risk Score (based on average age per disease)
    disease_age_avg = dfd.groupby('Disease')['Age'].mean()
    dfd['Risk_Score'] = dfd['Disease'].map(disease_age_avg)

    # Age squared
    dfd['Age_Squared'] = dfd['Age'] ** 2

    # Age group
    dfd['Age_Group'] = pd.cut(dfd['Age'], bins=[0, 35, 60, 100], labels=['Young', 'Adult', 'Elderly'])

    # One-hot encoding for categorical variables
    categorical_cols = ['Fever', 'Cough', 'Fatigue', 'DB', 'BP', 'CL', 'Gender']
    for col in categorical_cols:
        dummies = pd.get_dummies(dfd[col], prefix=col)
        dfd = pd.concat([dfd, dummies], axis=1)

    # Label encode target variable
    le = LabelEncoder()
    dfd['Results_Encoded'] = le.fit_transform(dfd['Results'])

    return dfd

# -------------------------------
# 4. Prepare Final Dataset
# -------------------------------
def prepare_dataset(dfd):
    """
    Select final features and separate X and y.
    """
    feature_cols = [
        'Age', 'Fever_and_Cough', 'Fever_and_Fatigue', 'Fatigue_and_Cough',
        'Fever_and_Fatigue_and_Cough', 'Disease_Frequency', 'Risk_Score',
        'Age_Squared', 'Fever_Yes', 'Cough_Yes', 'Fatigue_Yes', 'DB_Yes',
        'BP_Low', 'BP_Normal', 'CL_Low', 'CL_Normal', 'Gender_Male',
        'Age_Group_Adult', 'Age_Group_Elderly'
    ]

    X = dfd[feature_cols]
    y = dfd['Results_Encoded']

    # Convert boolean columns to int
    bool_cols = X.select_dtypes(include='bool').columns
    X[bool_cols] = X[bool_cols].astype(int)

    return X, y

# -------------------------------
# 5. Train and Evaluate Model
# -------------------------------
def train_and_evaluate(X, y):
    """
    Split data, train Random Forest, and evaluate performance.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    return model, scaler

# -------------------------------
# 6. Main Execution
# -------------------------------
def main():
    print("Loading data...")
    df = load_data()

    print("\nCategorical summaries:")
    for col in ['Gender', 'Fatigue', 'DB']:
        cat_summary(df, col, plot=True)

    print("\nNumerical summaries:")
    num_summary(df, 'Age', plot=True)

    print("\nCreating features...")
    dfd = create_features(df)

    print("\nFinal dataset preview:")
    print(dfd.head())

    print("\nPreparing dataset for modeling...")
    X, y = prepare_dataset(dfd)

    print("Feature matrix shape:", X.shape)
    print("Target distribution:\n", y.value_counts())

    print("\nTraining and evaluating model...")
    model, scaler = train_and_evaluate(X, y)

    print("\n✅ Model training complete.")

if __name__ == "__main__":
    main()