import re
import json
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.inspection import permutation_importance

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False


# =========================
# Configuration
# =========================

CSV_PATH = "synthetic_diagnostic_hallucination_dataset_1000_unique_complaints.csv"
RANDOM_STATE = 42

# Main decision tree
MAX_DEPTH = 5
MIN_SAMPLES_LEAF = 10

# Train/test split: 200 train / 800 test
TRAIN_SIZE = 200
TEST_SIZE = 800

# Optional model save paths
SAVE_MODELS = False
TREE_MODEL_PATH = "hallucination_decision_tree.joblib"
FOREST_MODEL_PATH = "hallucination_random_forest.joblib"


# =========================
# Feature extraction vocab
# =========================

LOW_SPECIFICITY_HINTS = {
    "meals", "meal", "spicy food", "wine", "stress", "anxiety", "panic",
    "family history", "overweight", "body aches", "dehydration",
    "subjective", "indigestion", "lifestyle", "cramping", "soreness",
    "food poisoning", "leftover antibiotics", "just reflux", "simply",
    "fatigue from stress", "body strain", "minor infection", "subjective fullness"
}

HIGH_SPECIFICITY_HINTS = {
    "radiation", "arm radiation", "jaw", "diaphoresis", "sweating", "exertional",
    "unilateral calf swelling", "calf swelling", "worsening dyspnea", "dyspnea",
    "morning stiffness", "symmetric small-joint", "small-joint", "rlq",
    "right lower quadrant", "movement pain", "weight loss", "tremor", "eye changes",
    "vomiting", "flank", "fever", "polyuria", "polydipsia",
    "blurry vision", "ketosis", "ascending from feet", "balance in dark",
    "cardiac risk", "diabetes", "goiter", "tachycardia", "systemic illness",
    "peritoneal irritation", "left arm", "climbing stairs", "palpitations"
}

RED_FLAG_HINTS = {
    "radiation", "arm", "jaw", "sweating", "diaphoresis", "cardiac risk",
    "exertional", "unilateral calf", "swelling", "worsening dyspnea",
    "eye changes", "weight loss", "vomiting", "fever", "flank", "ketosis",
    "rlq", "systemic illness", "diabetes", "left arm", "climbing stairs"
}

OVERCONFIDENT_HINTS = {
    "confirms", "must be", "best explanation", "simple diagnosis",
    "straightforward", "does not matter", "incidental", "only",
    "is primary", "explains all symptoms", "better treated as",
    "definitely", "clearly", "obviously"
}

DISMISSIVE_LABEL_HINTS = {
    "ignore", "downplay", "dismiss", "dismissal", "reinterpret", "reinterpretation",
    "overclaim", "generalization", "claim"
}

GROUNDING_LABEL_HINTS = {
    "contrast", "background", "elaboration", "support"
}


# =========================
# Text normalization helpers
# =========================

def normalize_text(text: str) -> str:
    return str(text).strip().lower()


def extract_segments(discourse_tree: str) -> List[Tuple[str, str]]:
    """
    Extract bracketed segments like:
      [Nucleus: ...]
      [Satellite-downplay: ...]
      [Conclusion: ...]
    """
    discourse_tree = normalize_text(discourse_tree)
    segments = re.findall(r"\[([^\]:]+):\s*([^\]]+)\]", discourse_tree)
    return [(label.strip(), text.strip()) for label, text in segments]


def get_root_text(discourse_tree: str) -> str:
    discourse_tree = normalize_text(discourse_tree)
    m = re.search(r"root:\s*([^.]+)\.", discourse_tree)
    return m.group(1).strip() if m else ""


def count_matches(text: str, vocab: set) -> int:
    text = normalize_text(text)
    return sum(1 for term in vocab if term in text)


def has_any(text: str, vocab: set) -> int:
    return int(count_matches(text, vocab) > 0)


# =========================
# Feature extraction
# =========================

def extract_tree_features(discourse_tree: str) -> Dict[str, float]:
    raw = normalize_text(discourse_tree)
    root_text = get_root_text(raw)
    segments = extract_segments(raw)

    labels = [label for label, _ in segments]
    texts = [text for _, text in segments]

    nucleus_texts = [text for label, text in segments if "nucleus" in label]
    satellite_texts = [text for label, text in segments if "satellite" in label]
    conclusion_texts = [text for label, text in segments if "conclusion" in label]

    nucleus_joined = " ".join(nucleus_texts)
    satellite_joined = " ".join(satellite_texts)
    conclusion_joined = " ".join(conclusion_texts)
    all_joined = " ".join(texts)

    # Label-based features
    has_ignore = int(any("ignore" in lbl for lbl in labels))
    has_downplay = int(any("downplay" in lbl for lbl in labels))
    has_dismissal = int(any("dismiss" in lbl for lbl in labels))
    has_overclaim = int(any("overclaim" in lbl for lbl in labels))
    has_reinterpretation = int(any("reinterpret" in lbl for lbl in labels))
    has_generalization = int(any("generalization" in lbl or "generalize" in lbl for lbl in labels))
    has_contrast = int(any("contrast" in lbl for lbl in labels))
    has_background = int(any("background" in lbl for lbl in labels))
    has_elaboration = int(any("elaboration" in lbl for lbl in labels))
    has_support = int(any("support" in lbl for lbl in labels))

    # Structural counts
    num_segments = len(segments)
    num_nucleus = sum(1 for lbl in labels if "nucleus" in lbl)
    num_satellite = sum(1 for lbl in labels if "satellite" in lbl)
    num_conclusion = sum(1 for lbl in labels if "conclusion" in lbl)

    num_dismissive_satellites = sum(
        1 for lbl in labels
        if ("satellite" in lbl and any(h in lbl for h in DISMISSIVE_LABEL_HINTS))
    )

    num_grounding_relations = sum(
        1 for lbl in labels
        if any(h in lbl for h in GROUNDING_LABEL_HINTS)
    )

    # Lexical clue features
    low_specificity_nucleus_count = count_matches(nucleus_joined, LOW_SPECIFICITY_HINTS)
    high_specificity_nucleus_count = count_matches(nucleus_joined, HIGH_SPECIFICITY_HINTS)
    low_specificity_satellite_count = count_matches(satellite_joined, LOW_SPECIFICITY_HINTS)
    high_specificity_satellite_count = count_matches(satellite_joined, HIGH_SPECIFICITY_HINTS)

    low_specificity_all_count = count_matches(all_joined, LOW_SPECIFICITY_HINTS)
    high_specificity_all_count = count_matches(all_joined, HIGH_SPECIFICITY_HINTS)

    root_mentions_favor = int("favor" in root_text)
    root_favors_low_specificity = int(low_specificity_nucleus_count > 0 and high_specificity_nucleus_count == 0)
    contradiction_preserved = int(has_contrast or has_background)
    conclusion_overconfident = has_any(conclusion_joined, OVERCONFIDENT_HINTS)
    red_flag_in_satellite = has_any(satellite_joined, RED_FLAG_HINTS)
    red_flag_ignored = int(red_flag_in_satellite and num_dismissive_satellites > 0)

    # Balance features
    nucleus_specificity_margin = high_specificity_nucleus_count - low_specificity_nucleus_count
    satellite_specificity_margin = high_specificity_satellite_count - low_specificity_satellite_count
    all_specificity_margin = high_specificity_all_count - low_specificity_all_count

    # Ratio features for greater sensitivity
    total_specificity_nucleus = high_specificity_nucleus_count + low_specificity_nucleus_count
    total_specificity_satellite = high_specificity_satellite_count + low_specificity_satellite_count
    total_specificity_all = high_specificity_all_count + low_specificity_all_count

    nucleus_high_specificity_ratio = (
        high_specificity_nucleus_count / total_specificity_nucleus
        if total_specificity_nucleus > 0 else 0.0
    )

    satellite_high_specificity_ratio = (
        high_specificity_satellite_count / total_specificity_satellite
        if total_specificity_satellite > 0 else 0.0
    )

    all_high_specificity_ratio = (
        high_specificity_all_count / total_specificity_all
        if total_specificity_all > 0 else 0.0
    )

    dismissive_satellite_ratio = (
        num_dismissive_satellites / num_satellite
        if num_satellite > 0 else 0.0
    )

    grounding_relation_ratio = (
        num_grounding_relations / num_segments
        if num_segments > 0 else 0.0
    )

    nucleus_share_of_segments = (
        num_nucleus / num_segments
        if num_segments > 0 else 0.0
    )

    satellite_share_of_segments = (
        num_satellite / num_segments
        if num_segments > 0 else 0.0
    )

    # Heuristic summary score
    heuristic_hallucination_score = (
        3 * has_ignore
        + 2 * has_downplay
        + 2 * has_dismissal
        + 2 * has_overclaim
        + 2 * has_reinterpretation
        + 1 * has_generalization
        + 2 * num_dismissive_satellites
        + 2 * root_favors_low_specificity
        + 3 * red_flag_ignored
        + 1 * conclusion_overconfident
        - 2 * has_contrast
        - 1 * has_background
        - 1 * has_elaboration
        - 1 * has_support
    )

    return {
        "has_ignore": has_ignore,
        "has_downplay": has_downplay,
        "has_dismissal": has_dismissal,
        "has_overclaim": has_overclaim,
        "has_reinterpretation": has_reinterpretation,
        "has_generalization": has_generalization,
        "has_contrast": has_contrast,
        "has_background": has_background,
        "has_elaboration": has_elaboration,
        "has_support": has_support,
        "num_segments": num_segments,
        "num_nucleus": num_nucleus,
        "num_satellite": num_satellite,
        "num_conclusion": num_conclusion,
        "num_dismissive_satellites": num_dismissive_satellites,
        "num_grounding_relations": num_grounding_relations,
        "low_specificity_nucleus_count": low_specificity_nucleus_count,
        "high_specificity_nucleus_count": high_specificity_nucleus_count,
        "low_specificity_satellite_count": low_specificity_satellite_count,
        "high_specificity_satellite_count": high_specificity_satellite_count,
        "low_specificity_all_count": low_specificity_all_count,
        "high_specificity_all_count": high_specificity_all_count,
        "root_mentions_favor": root_mentions_favor,
        "root_favors_low_specificity": root_favors_low_specificity,
        "contradiction_preserved": contradiction_preserved,
        "conclusion_overconfident": conclusion_overconfident,
        "red_flag_in_satellite": red_flag_in_satellite,
        "red_flag_ignored": red_flag_ignored,
        "nucleus_specificity_margin": nucleus_specificity_margin,
        "satellite_specificity_margin": satellite_specificity_margin,
        "all_specificity_margin": all_specificity_margin,
        "nucleus_high_specificity_ratio": nucleus_high_specificity_ratio,
        "satellite_high_specificity_ratio": satellite_high_specificity_ratio,
        "all_high_specificity_ratio": all_high_specificity_ratio,
        "dismissive_satellite_ratio": dismissive_satellite_ratio,
        "grounding_relation_ratio": grounding_relation_ratio,
        "nucleus_share_of_segments": nucleus_share_of_segments,
        "satellite_share_of_segments": satellite_share_of_segments,
        "heuristic_hallucination_score": heuristic_hallucination_score,
    }


# =========================
# Dataset loading
# =========================

def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = {
        "discourse_tree_of_reasoning_log",
        "hallucinated_in_diagnosis_making",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df["hallucinated_label"] = (
        df["hallucinated_in_diagnosis_making"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )

    if df["hallucinated_label"].isna().any():
        bad_values = df.loc[
            df["hallucinated_label"].isna(),
            "hallucinated_in_diagnosis_making"
        ].unique()
        raise ValueError(f"Unexpected hallucination labels: {bad_values}")

    return df


def build_feature_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    feature_rows = df["discourse_tree_of_reasoning_log"].apply(extract_tree_features)
    return pd.DataFrame(list(feature_rows))


# =========================
# Training
# =========================

def train_decision_tree(
    X: pd.DataFrame,
    y: pd.Series
) -> Tuple[DecisionTreeClassifier, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=TRAIN_SIZE,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=MAX_DEPTH,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)
    return clf, X_train, X_test, y_train, y_test


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    return rf


# =========================
# Reporting
# =========================

def print_split_sizes(X_train, X_test, y_train, y_test) -> None:
    print("\n=== Split sizes ===")
    print(f"Train size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")
    print(f"Train hallucination rate: {y_train.mean():.3f}")
    print(f"Test hallucination rate:  {y_test.mean():.3f}")


def print_evaluation(model, X_test: pd.DataFrame, y_test: pd.Series, title: str) -> None:
    y_pred = model.predict(X_test)

    print(f"\n=== {title}: Accuracy ===")
    print(accuracy_score(y_test, y_pred))

    print(f"\n=== {title}: F1 ===")
    print(f1_score(y_test, y_pred))

    print(f"\n=== {title}: Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    print(f"\n=== {title}: Classification Report ===")
    print(classification_report(
        y_test,
        y_pred,
        target_names=["not_hallucination", "hallucination"]
    ))


def print_tree_structure(clf: DecisionTreeClassifier, X: pd.DataFrame) -> None:
    print("\n=== Learned Decision Tree ===")
    print(export_text(clf, feature_names=list(X.columns)))


def print_impurity_importance(model, X: pd.DataFrame, title: str, top_k: int = 20) -> None:
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "impurity_importance": model.feature_importances_
    }).sort_values("impurity_importance", ascending=False)

    print(f"\n=== {title}: Impurity-based Feature Importances ===")
    print(importance_df.head(top_k).to_string(index=False))


def print_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    title: str,
    scoring: str = "f1",
    n_repeats: int = 30,
    top_k: int = 20
) -> None:
    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=-1
    )

    perm_df = pd.DataFrame({
        "feature": X_test.columns,
        "perm_mean": perm.importances_mean,
        "perm_std": perm.importances_std
    }).sort_values("perm_mean", ascending=False)

    print(f"\n=== {title}: Permutation Importances ({scoring}) ===")
    print(perm_df.head(top_k).to_string(index=False))


def print_feature_correlation_summary(X: pd.DataFrame, top_k: int = 20) -> None:
    """
    Useful for diagnosing why one feature may dominate:
    some features are near-duplicates.
    """
    corr = X.corr(numeric_only=True).abs()
    pairs = []

    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i, j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    print("\n=== Highly correlated feature pairs ===")
    for a, b, c in pairs[:top_k]:
        print(f"{a:35s} {b:35s} {c:.3f}")


# =========================
# Inference
# =========================

def classify_new_tree(model, discourse_tree: str, feature_columns: List[str]) -> Dict:
    features = extract_tree_features(discourse_tree)
    X_new = pd.DataFrame([features])[feature_columns]

    pred = int(model.predict(X_new)[0])
    prob = model.predict_proba(X_new)[0].tolist()

    return {
        "prediction_label": "hallucination" if pred == 1 else "not_hallucination",
        "prediction_numeric": pred,
        "probabilities": {
            "not_hallucination": prob[0],
            "hallucination": prob[1],
        },
        "features": features,
    }


# =========================
# Optional save
# =========================

def maybe_save_model(model, path: str) -> None:
    if SAVE_MODELS:
        if not JOBLIB_AVAILABLE:
            print(f"joblib not installed; could not save {path}")
            return
        joblib.dump(model, path)
        print(f"Saved model to {path}")


# =========================
# Main
# =========================

def main():
    print("Loading dataset...")
    df = load_dataset(CSV_PATH)

    print("Extracting discourse-tree features...")
    X = build_feature_dataframe(df)
    y = df["hallucinated_label"]

    print("Training main decision tree...")
    clf, X_train, X_test, y_train, y_test = train_decision_tree(X, y)

    print_split_sizes(X_train, X_test, y_train, y_test)
    print_evaluation(clf, X_test, y_test, title="DecisionTree")
    print_impurity_importance(clf, X_train, title="DecisionTree", top_k=25)
    print_permutation_importance(clf, X_test, y_test, title="DecisionTree", scoring="f1", n_repeats=30, top_k=25)
    print_tree_structure(clf, X)
    print_feature_correlation_summary(X, top_k=15)

    print("\nTraining RandomForest for more stable importances...")
    rf = train_random_forest(X_train, y_train)
    print_evaluation(rf, X_test, y_test, title="RandomForest")
    print_impurity_importance(rf, X_train, title="RandomForest", top_k=25)
    print_permutation_importance(rf, X_test, y_test, title="RandomForest", scoring="f1", n_repeats=20, top_k=25)

    maybe_save_model(clf, TREE_MODEL_PATH)
    maybe_save_model(rf, FOREST_MODEL_PATH)

    # Example inference
    example_tree_valid = (
        "Root: favor ACS. "
        "[Nucleus: exertional pain + radiation + sweating -> ischemia] "
        "[Satellite-contrast: spicy food/nighttime + antacid relief -> GERD-like] "
        "[Nucleus-elaboration: age + diabetes increase cardiac risk] "
        "[Conclusion: prioritize ACS]"
    )

    example_tree_hall = (
        "Root: favor GERD. "
        "[Nucleus: spicy food + antacid relief -> reflux] "
        "[Satellite-downplay: arm radiation + sweating -> anxiety response] "
        "[Satellite-ignore: exertional trigger + diabetes] "
        "[Conclusion: GERD explains symptoms]"
    )

    print("\n=== Example: valid-like discourse tree (DecisionTree) ===")
    result_valid = classify_new_tree(clf, example_tree_valid, list(X.columns))
    print(json.dumps(result_valid, indent=2))

    print("\n=== Example: hallucination-like discourse tree (DecisionTree) ===")
    result_hall = classify_new_tree(clf, example_tree_hall, list(X.columns))
    print(json.dumps(result_hall, indent=2))

    print("\n=== Example: valid-like discourse tree (RandomForest) ===")
    result_valid_rf = classify_new_tree(rf, example_tree_valid, list(X.columns))
    print(json.dumps(result_valid_rf, indent=2))

    print("\n=== Example: hallucination-like discourse tree (RandomForest) ===")
    result_hall_rf = classify_new_tree(rf, example_tree_hall, list(X.columns))
    print(json.dumps(result_hall_rf, indent=2))


if __name__ == "__main__":
    main()