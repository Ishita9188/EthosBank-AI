import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)

from sklearn.model_selection import train_test_split


# ============================================================
# ETHOSBANK AI - OFFER SCAN MODEL TRAINING V3
# ============================================================

print("\n" + "=" * 75)
print("ETHOSBANK AI - OFFER SCAN MODEL TRAINING VERSION 3")
print("=" * 75)

print("\nIMPORTANT:")
print("Version 1 & 2 models will NOT be overwritten.")
print("New models will be saved with '_v3.pkl' filenames.")


# ============================================================
# PATH SETUP
# ============================================================

print("\n[1/8] Setting up project paths...")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODEL_DIR = os.path.join(CURRENT_DIR, "ml_models")
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Project root: {PROJECT_ROOT}")
print(f"Model directory: {MODEL_DIR}")


# ============================================================
# IMPORT PREPROCESSING
# ============================================================

print("\n[2/8] Importing preprocessing module...")

from offer_scan.preprocess import prepare_offer_scan_dataset

print("Preprocessing module imported successfully.")


# ============================================================
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model, X_test, y_test, model_name):

    print("\n" + "-" * 75)
    print(f"EVALUATING: {model_name}")
    print("-" * 75)

    print("\n[PREDICTION] Generating predictions...")
    predictions = model.predict(X_test)
    print("[PREDICTION] Completed.")

    accuracy = accuracy_score(y_test, predictions)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )

    print("\n" + "=" * 75)
    print("MODEL PERFORMANCE")
    print("=" * 75)

    print(f"\nAccuracy:           {accuracy * 100:.2f}%")
    print(f"Weighted Precision: {precision * 100:.2f}%")
    print(f"Weighted Recall:    {recall * 100:.2f}%")
    print(f"Weighted F1 Score:  {f1 * 100:.2f}%")

    print("\nMacro Average Metrics:")
    print(f"Macro Precision:    {macro_precision * 100:.2f}%")
    print(f"Macro Recall:       {macro_recall * 100:.2f}%")
    print(f"Macro F1 Score:     {macro_f1 * 100:.2f}%")

    print("\nClassification Report:\n")
    print(classification_report(y_test, predictions, zero_division=0))

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }


# ============================================================
# SAVE MODEL
# ============================================================

def save_model(model, filename):

    model_path = os.path.join(MODEL_DIR, filename)

    print("\n[SAVING] Saving Version 3 model...")
    print(f"New model file:\n{model_path}")

    joblib.dump(model, model_path)

    print(f"[SUCCESS] Model saved as {filename}")

    return model_path


# ============================================================
# FINANCIAL SENTIMENT V3
# ============================================================

def train_financial_sentiment_v3(phrasebank_df):

    print("\n" + "#" * 75)
    print("MODEL 1 OF 4: FINANCIAL SENTIMENT MODEL V3")
    print("#" * 75)

    X = phrasebank_df["text"]
    y = phrasebank_df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True,
                min_df=1,
                max_df=0.98,
                max_features=30000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=5000,
                C=2.0,
                class_weight="balanced",
                solver="lbfgs"
            )
        )
    ])

    print("\n[TRAINING] Training Financial Sentiment V3...")
    model.fit(X_train, y_train)
    print("[TRAINING] Completed.")

    metrics = evaluate_model(model, X_test, y_test, "financial_sentiment_model_v3")
    save_model(model, "financial_sentiment_model_v3.pkl")

    return metrics


# ============================================================
# BANKING INTENT V3
# ============================================================

def train_banking_intent_v3(train_df, test_df):

    print("\n" + "#" * 75)
    print("MODEL 2 OF 4: BANKING INTENT MODEL V3")
    print("#" * 75)

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        sublinear_tf=True,
        min_df=1,
        max_features=50000
    )

    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        max_features=50000
    )

    combined_features = ColumnTransformer([
        ("word_features", word_vectorizer, "text"),
        ("char_features", char_vectorizer, "text")
    ])

    X_train_df = X_train.to_frame(name="text")
    X_test_df = X_test.to_frame(name="text")

    model = Pipeline([
        ("features", combined_features),
        ("classifier", LinearSVC(C=1.5, class_weight="balanced", random_state=42))
    ])

    print("\n[TRAINING] Training Banking Intent V3...")
    model.fit(X_train_df, y_train)
    print("[TRAINING] Completed.")

    metrics = evaluate_model(model, X_test_df, y_test, "banking_intent_model_v3")
    save_model(model, "banking_intent_model_v3.pkl")

    return metrics


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

NUMERIC_FEATURES = [
    "interest_rate",
    "processing_fee",
    "loan_tenure",
    "contains_terms_and_conditions",
    "mentions_processing_fee",
    "mentions_interest_rate",
    "mentions_penalty",
    "mentions_foreclosure",
    "urgency_phrase",
    "scarcity_phrase",
    "emotional_phrase",
    "hidden_fee_indicator",
    "transparency_score",
    "manipulation_score",
    "complexity_score"
]

CATEGORICAL_FEATURES = [
    "institution_type",
    "product",
    "primary_trigger",
    "secondary_trigger",
    "offer_category"
]


def clean_numeric_features(df):

    df = df.copy()

    for column in NUMERIC_FEATURES:
        if column not in df.columns:
            continue

        df[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.replace("%", "", regex=False)
            .str.replace("₹", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        df[column] = df[column].replace(
            {"": np.nan, "nan": np.nan, "None": np.nan, "N/A": np.nan, "NA": np.nan, "-": np.nan}
        )

        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def create_offer_feature_processor():

    text_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), sublinear_tf=True, max_features=20000))
    ])

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    processor = ColumnTransformer([
        ("text", text_pipeline, "text"),
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return processor


# ============================================================
# ETHICAL OFFER MODEL V3
# ============================================================

def train_ethical_offer_v3(custom_df):

    print("\n" + "#" * 75)
    print("MODEL 3 OF 4: ETHICAL OFFER MODEL V3")
    print("#" * 75)

    clean_df = clean_numeric_features(custom_df)

    X = clean_df[["text"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = clean_df["ethical_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    feature_processor = create_offer_feature_processor()

    model = Pipeline([
        ("features", feature_processor),
        ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ])

    print("\n[TRAINING] Training Ethical Offer Model V3 (Random Forest)...")
    model.fit(X_train, y_train)
    print("[TRAINING] Completed.")

    metrics = evaluate_model(model, X_test, y_test, "ethical_offer_model_v3")
    save_model(model, "ethical_offer_model_v3.pkl")

    return metrics


# ============================================================
# OFFER RISK MODEL V3
# ============================================================

def train_offer_risk_v3(custom_df):

    print("\n" + "#" * 75)
    print("MODEL 4 OF 4: OFFER RISK MODEL V3")
    print("#" * 75)

    clean_df = clean_numeric_features(custom_df)

    X = clean_df[["text"] + NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = clean_df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    feature_processor = create_offer_feature_processor()

    model = Pipeline([
        ("features", feature_processor),
        ("classifier", RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42))
    ])

    print("\n[TRAINING] Training Offer Risk Model V3 (Random Forest)...")
    model.fit(X_train, y_train)
    print("[TRAINING] Completed.")

    metrics = evaluate_model(model, X_test, y_test, "offer_risk_model_v3")
    save_model(model, "offer_risk_model_v3.pkl")

    return metrics


# ============================================================
# MAIN V3 TRAINING PIPELINE
# ============================================================

def train_all_models_v3():

    data = prepare_offer_scan_dataset()

    phrasebank_df = data["phrasebank"]
    banking77_train = data["banking77_train"]
    banking77_test = data["banking77_test"]
    custom_df = data["custom"]

    sentiment_metrics = train_financial_sentiment_v3(phrasebank_df)
    intent_metrics = train_banking_intent_v3(banking77_train, banking77_test)
    ethical_metrics = train_ethical_offer_v3(custom_df)
    risk_metrics = train_offer_risk_v3(custom_df)

    print("\n" + "=" * 75)
    print("[8/8] VERSION 3 TRAINING COMPLETE")
    print("=" * 75)

    print(f"\nFinancial Sentiment Accuracy: {sentiment_metrics['accuracy'] * 100:.2f}%")
    print(f"Banking Intent Accuracy:      {intent_metrics['accuracy'] * 100:.2f}%")
    print(f"Ethical Offer Accuracy:       {ethical_metrics['accuracy'] * 100:.2f}%")
    print(f"Offer Risk Accuracy:          {risk_metrics['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    train_all_models_v3()