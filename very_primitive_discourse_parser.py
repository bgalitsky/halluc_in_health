import spacy

# Load spaCy model (download first if needed: python -m spacy download en_core_web_sm)
nlp = spacy.load("en_core_web_sm")


def discourse_parse(complaint):
    """
    Very simplified discourse parsing:
    - Split text into sentences
    - Classify sentence roles: BACKGROUND, SYMPTOM, IMPACT, WORRY
    """
    doc = nlp(complaint)
    results = []

    for sent in doc.sents:
        text = sent.text.strip()
        label = "OTHER"

        # Rule-based discourse classification
        if any(word in text.lower() for word in ["i am", "as a", "years old"]):
            label = "BACKGROUND"
        elif any(word in text.lower() for word in
                 ["fever", "cough", "fatigue", "pain", "short of breath", "difficulty breathing"]):
            label = "SYMPTOM"
        elif any(word in text.lower() for word in
                 ["harder", "can’t", "difficult", "struggling", "exhausted", "drained"]):
            label = "IMPACT"
        elif any(word in text.lower() for word in ["worried", "concerned", "afraid", "suspect", "might be"]):
            label = "WORRY"

        results.append({"sentence": text, "label": label})

    return results


# Example medical complaint
complaint = ("I am a 19-year-old female, and lately I have been dealing with fever and fatigue for the past few days. "
             "It makes everyday activities, like walking or working, much harder. "
             "I’m concerned it might be related to influenza. "
             "My blood pressure is low and cholesterol is normal.")

parsed = discourse_parse(complaint)

for seg in parsed:
    print(f"[{seg['label']}] {seg['sentence']}")
