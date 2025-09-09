import pandas as pd
import random

# Load dataset
df = pd.read_csv("Disease_symptom_and_patient_profile_dataset.csv")


def generate_detailed_complaint(row):
    disease = row["Disease"]
    age, gender = row["Age"], row["Gender"].lower()

    # Collect symptoms
    symptoms = []
    for col in ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]:
        if str(row[col]).lower() in ["yes", "1"]:
            symptoms.append(col.lower())
    if not symptoms:
        symptoms_text = "a general feeling of being unwell"
    else:
        symptoms_text = ", ".join(symptoms)

    # Variations
    durations = [
        "for the past three days",
        "for almost a week now",
        "since last weekend",
        "gradually worsening over the past two weeks"
    ]
    progressions = [
        "It started mildly but has been getting worse each day.",
        "The symptoms come and go, but overall they are becoming more frequent.",
        "It began suddenly and has stayed intense since then.",
        "At first it was tolerable, but now it is interfering with almost everything I do."
    ]
    impacts = [
        "I have trouble sleeping through the night and wake up feeling exhausted.",
        "Simple tasks like climbing stairs or preparing meals leave me drained.",
        "I have had to miss work and cancel plans because I feel too weak.",
        "It’s affecting my concentration and I can’t keep up with daily responsibilities."
    ]
    concerns = [
        f"I am worried it might be {disease.lower()} because the symptoms are so persistent.",
        f"My family has a history of {disease.lower()}, so I am particularly concerned.",
        f"This doesn’t feel like a regular cold; it seems more like {disease.lower()}.",
        f"I suspect it could be {disease.lower()}, but I’m unsure and it worries me."
    ]

    complaint = (
        f"I am a {age}-year-old {gender}. I have been experiencing {symptoms_text} "
        f"{random.choice(durations)}. {random.choice(progressions)} "
        f"{random.choice(impacts)} {random.choice(concerns)} "
        f"My blood pressure is {row['Blood Pressure'].lower()} and my cholesterol level is {row['Cholesterol Level'].lower()}."
    )

    return complaint


# Apply to dataset
df["Detailed_Complaint"] = df.apply(generate_detailed_complaint, axis=1)

# Save
output_path = "disease_complaints_detailed.csv"
df.to_csv(output_path, index=False)

print(f"✅ Detailed complaints saved to {output_path}")
