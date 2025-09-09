import pandas as pd
from openai import OpenAI
import json
from configparser import RawConfigParser
import os

#https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset

config = RawConfigParser()
config.read('config.ini')

# API keys
gemini_api_key = config.get('Gemini', 'api_key')
os.environ["OPENAI_API_KEY"] = config.get('OpenAI', 'api_key')
# Initialize OpenAI client
client = OpenAI(api_key=config.get('OpenAI', 'api_key'))

# Load the dataset with imaginative complaints
df = pd.read_csv("data/Symptom2Disease_Extended_Fragment.csv")

def diagnose_with_gpt(complaint: str):
    """
    Uses GPT-4o to return:
    - diagnosis (one phrase)
    - explanation (short reasoning)
    """
    prompt = f"""
    Read the following patient health complaint:

    "{complaint}"

    Provide two outputs:
    1. The most likely diagnosis in ONE phrase only (e.g., "Influenza", "Asthma", "Eczema").
    2. A short explanation (2–3 sentences) of why this diagnosis fits the complaint.

    Return the answer in strict JSON format:
    {{
      "diagnosis": "...",
      "explanation": "..."
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=1
    )

    content = response.choices[0].message.content

    try:
        result = json.loads(content)
    except:
        # fallback if parsing fails
        result = {"diagnosis": "Unknown", "explanation": content}

    return result["diagnosis"], result["explanation"]

# Iterate through dataset
pred_diagnoses = []
explanations = []
matches = []

count =0
for complaint, true_disease in zip(df["Detailed Complaint"], df["Disease"]):
    diagnosis, explanation = diagnose_with_gpt(complaint)
    pred_diagnoses.append(diagnosis)
    explanations.append(explanation)
    matches.append(diagnosis.lower() == str(true_disease).lower())
    count=count+1
    #if count>3:
    #    break
    print(str(true_disease).lower())

# Add results to DataFrame
df["Predicted_Diagnosis"] = pred_diagnoses
df["Explanation"] = explanations
df["Match_with_Database"] = matches

# Save to new file
output_path = "results/detailed_complaints_with_diagnosis_fragment.csv"
df.to_csv(output_path, index=False)

print(f"✅ Results saved to {output_path}")
