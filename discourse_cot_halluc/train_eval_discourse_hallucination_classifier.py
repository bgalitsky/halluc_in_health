import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import json

DATA_PATH = Path("blinded_discourse_dataset_500_500.csv")  # edit path if needed
df = pd.read_csv(DATA_PATH)

label_map = {"correct": 0, "hallucinated": 1}
y = df["label"].map(label_map)

def train_eval(text_col: str, name: str):
    X = df[text_col].fillna("")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    clf = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True
        )),
        ("logreg", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="liblinear"
        ))
    ])

    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        clf, X, y, cv=cv,
        scoring=["accuracy", "f1", "precision", "recall"]
    )

    report = classification_report(
        y_test, pred,
        target_names=["correct", "hallucinated"],
        output_dict=True,
        zero_division=0
    )

    return {
        "name": name,
        "text_column": text_col,
        "holdout_accuracy": accuracy_score(y_test, pred),
        "holdout_f1_hallucinated": f1_score(y_test, pred),
        "holdout_confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        "holdout_report": report,
        "cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
        "cv_accuracy_std": cv_scores["test_accuracy"].std(),
        "cv_f1_mean": cv_scores["test_f1"].mean(),
        "cv_f1_std": cv_scores["test_f1"].std(),
        "cv_precision_mean": cv_scores["test_precision"].mean(),
        "cv_recall_mean": cv_scores["test_recall"].mean(),
    }

results = [
    train_eval("custom_discourse_tree_blinded", "Custom discourse tree only"),
    train_eval("traditional_rst_tree_blinded", "Traditional RST tree only"),
]

print(json.dumps(results, indent=2))
