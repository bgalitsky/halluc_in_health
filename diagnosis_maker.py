import pandas as pd
from openai import OpenAI
import json
from configparser import RawConfigParser
import os

config = RawConfigParser()
config.read('config.ini')

# API keys
gemini_api_key = config.get('Gemini', 'api_key')
os.environ["OPENAI_API_KEY"] = config.get('OpenAI', 'api_key')
# Initialize OpenAI client
client = OpenAI(api_key=config.get('OpenAI', 'api_key'))

# Load the dataset with imaginative complaints
df = pd.read_csv("complaints_dataset_fragment.csv")

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
        temperature=0.3
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

for complaint, true_disease in zip(df["Imaginative_Complaint"], df["Disease"]):
    diagnosis, explanation = diagnose_with_gpt(complaint)
    pred_diagnoses.append(diagnosis)
    explanations.append(explanation)
    matches.append(diagnosis.lower() == str(true_disease).lower())

# Add results to DataFrame
df["Predicted_Diagnosis"] = pred_diagnoses
df["Explanation"] = explanations
df["Match_with_Database"] = matches

# Save to new file
output_path = "complaints_with_diagnosis.csv"
df.to_csv(output_path, index=False)

print(f"✅ Results saved to {output_path}")
