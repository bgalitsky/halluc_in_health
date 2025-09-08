import os
from configparser import RawConfigParser
import pandas as pd
import openai

# Load your dataset
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")

# Load configuration
config = RawConfigParser()
config.read('config.ini')

# API keys
gemini_api_key = config.get('Gemini', 'api_key')
os.environ["OPENAI_API_KEY"] = config.get('OpenAI', 'api_key')

def polish_complaint_with_gpt(raw_complaint):
    prompt = f"""
    Take the following short medical note and turn it into a longer, imaginative health complaint
    written in natural patient language. Add realistic details such as duration of symptoms, how
    they affect daily activities, any worries the patient may have, and emotional tone. Keep it
    sounding plausible, like something a patient would say to a doctor.

    Original note:
    "{raw_complaint}"
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.9,  # more creative
        max_tokens=250
    )
    return response.choices[0].message["content"]

# Function to convert one row into a health complaint text
def make_complaint(row):
    # Concatenate symptoms
    symptoms = []
    for col in df.columns:
        if row[col] == 1 or str(row[col]).lower() == "yes":
            symptoms.append(col)
    symptom_text = ", ".join(symptoms)

    # Base complaint
    complaint = f"The patient reports experiencing {symptom_text}. The condition appears to be related to {row.get('Disease', 'an unspecified illness')}."
    return complaint

# Example: take row 0
row = df.iloc[0]
raw_complaint = make_complaint(row)

# (Optional) Use ChatGPT to polish text further
openai.api_key = "YOUR_API_KEY"
complaint_text = polish_complaint_with_gpt(raw_complaint)
print(complaint_text)





# Example: polish the first row's complaint
row = df.iloc[0]
raw_complaint = make_complaint(row)
longer_complaint = polish_complaint_with_gpt(raw_complaint)

print("Raw:", raw_complaint)
print("\nImaginative complaint:\n", longer_complaint)
