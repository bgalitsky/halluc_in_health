import pandas as pd
import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
from configparser import RawConfigParser

#https://colab.research.google.com/#scrollTo=ndLRGovKrGFz&fileId=https%3A//huggingface.co/hafidev/bert-base-uncased-discourse-markers-disfluency-detection-beta-v1.ipynb
#https://pypi.org/project/edu-segmentation/

config = RawConfigParser()
config.read('config.ini')

# API keys
HF_TOKEN = config.get('HuggingFace', 'api_key')

# Download sentence tokenizer
nltk.download("punkt")

login(token=HF_TOKEN)  # this logs you into Hugging Face for this session

# --- Step 2: Load DisSent discourse parser ---
MODEL = "nyu-mll/dissent-roberta-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL, use_auth_token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, use_auth_token=HF_TOKEN)


def parse_relation(sent1, sent2):
    """Predict discourse relation between two consecutive sentences."""
    inputs = tokenizer(sent1, sent2, return_tensors="pt", truncation=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    label_id = torch.argmax(probs).item()
    label = model.config.id2label[label_id]
    confidence = probs[0][label_id].item()
    return label, confidence


def build_discourse_tree_and_extract(text, disease):
    """Split complaint into sentences, predict relations, extract diagnostic sentence."""
    sentences = nltk.sent_tokenize(text)
    relations = []
    diagnostic_sentence = None

    for i in range(len(sentences) - 1):
        rel, conf = parse_relation(sentences[i], sentences[i + 1])
        relations.append({
            "from": sentences[i],
            "to": sentences[i + 1],
            "relation": rel,
            "confidence": round(conf, 2)
        })

        # Rule 1: pick disease mention
        if disease.lower() in sentences[i].lower():
            diagnostic_sentence = sentences[i]

        # Rule 2: strong causal link
        if "CAUSE" in rel or "CONTINGENCY" in rel:
            diagnostic_sentence = sentences[i + 1]

    # Fallback: pick the longest sentence (often symptom-rich)
    if not diagnostic_sentence:
        diagnostic_sentence = max(sentences, key=len)

    return relations, diagnostic_sentence


# Load complaints dataset
df = pd.read_csv("disease_complaints_detailed.csv")

# Apply discourse parsing and diagnostic sentence extraction (on a sample)
df_sample = df.sample(5, random_state=42).copy()
df_sample["Results"] = df_sample.apply(
    lambda row: build_discourse_tree_and_extract(row["Detailed_Complaint"], row["Disease"]),
    axis=1
)

# Split into columns
df_sample["Discourse_Structure"] = df_sample["Results"].apply(lambda x: x[0])
df_sample["Diagnostic_Sentence"] = df_sample["Results"].apply(lambda x: x[1])

df_sample = df_sample[["Disease", "Detailed_Complaint", "Diagnostic_Sentence", "Discourse_Structure"]]
