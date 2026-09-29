import os
import sys
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ============================================================
# PROGRAM START
# ============================================================

print("\n" + "=" * 70)
print("ETHOSBANK AI - OFFER SCAN MODEL TRAINING")
print("=" * 70)

print("\n[1/10] Starting training program...")


# ============================================================
# ADD PROJECT ROOT TO PYTHON PATH
# ============================================================

print("\n[2/10] Setting up project paths...")

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    CURRENT_DIR
)

print(f"Current directory: {CURRENT_DIR}")
print(f"Project root:      {PROJECT_ROOT}")

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

print("Project root added to Python path.")


# ============================================================
# IMPORT PREPROCESSING FUNCTION
# ============================================================

print("\n[3/10] Importing preprocessing module...")

from offer_scan.preprocess import (
    prepare_offer_scan_dataset
)

print("Preprocessing module imported successfully.")


# ============================================================
# MODEL DIRECTORY
# ============================================================

print("\n[4/10] Preparing ML model directory...")

MODEL_DIR = os.path.join(
    CURRENT_DIR,
    "ml_models"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

print(f"Model directory ready: {MODEL_DIR}")


# ============================================================
# CREATE NLP PIPELINE
# ============================================================

def create_text_classifier():

    print("    Creating TF-IDF vectorizer...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=10000
    )

    print("    Creating Logistic Regression classifier...")

    classifier = LogisticRegression(
        max_iter=2000,
        class_weight="balanced"
    )

    print("    Combining TF-IDF + Logistic Regression pipeline...")

    pipeline = Pipeline(
        [
            ("tfidf", vectorizer),
            ("classifier", classifier)
        ]
    )

    print("    NLP pipeline created successfully.")

    return pipeline


# ============================================================
# TRAIN AND SAVE MODEL
# ============================================================

def train_and_save_model(
    X_train,
    X_test,
    y_train,
    y_test,
    model_name
):

    print("\n" + "=" * 70)
    print(f"STARTING MODEL: {model_name}")
    print("=" * 70)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")

    print("\nUnique classes:")

    for label in sorted(y_train.unique()):
        count = (y_train == label).sum()
        print(f"    {label}: {count} training samples")

    print("\n[TRAINING] Creating machine learning pipeline...")

    model = create_text_classifier()

    print("\n[TRAINING] Model pipeline ready.")

    print(
        "[TRAINING] Starting model training..."
    )

    print(
        "[TRAINING] Please wait. "
        "This may take some time depending on the dataset..."
    )

    model.fit(
        X_train,
        y_train
    )

    print(
        "[TRAINING] Model training completed successfully!"
    )

    print(
        "\n[PREDICTION] Running predictions on test data..."
    )

    predictions = model.predict(
        X_test
    )

    print(
        "[PREDICTION] Predictions completed."
    )

    print(
        "\n[EVALUATION] Calculating model accuracy..."
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nModel Accuracy: {accuracy:.4f}"
    )

    print(
        "\n[EVALUATION] Generating classification report..."
    )

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print(
        "\n[SAVING] Preparing model file..."
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"{model_name}.pkl"
    )

    print(
        f"[SAVING] Saving model to:\n{model_path}"
    )

    joblib.dump(
        model,
        model_path
    )

    print(
        f"[SUCCESS] {model_name} saved successfully!"
    )

    print("=" * 70)

    return model


# ============================================================
# 1. TRAIN FINANCIAL SENTIMENT MODEL
# ============================================================

def train_sentiment_model(
    phrasebank_df
):

    print("\n" + "#" * 70)
    print("MODEL 1 OF 4: FINANCIAL SENTIMENT MODEL")
    print("#" * 70)

    print(
        "\nPreparing Financial PhraseBank dataset..."
    )

    print(
        f"Total Financial PhraseBank records: "
        f"{len(phrasebank_df)}"
    )

    print(
        "\nAvailable columns:"
    )

    print(
        list(phrasebank_df.columns)
    )

    print(
        "\nExtracting input text..."
    )

    X = phrasebank_df[
        "text"
    ]

    print(
        "Extracting sentiment labels..."
    )

    y = phrasebank_df[
        "label"
    ]

    print(
        "\nSplitting data into training and testing sets..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print(
        "Dataset split completed."
    )

    print(
        f"Training data: {len(X_train)} records"
    )

    print(
        f"Testing data:  {len(X_test)} records"
    )

    train_and_save_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "financial_sentiment_model"
    )

    print(
        "\nFINANCIAL SENTIMENT MODEL COMPLETED."
    )


# ============================================================
# 2. TRAIN BANKING INTENT MODEL
# ============================================================

def train_banking_intent_model(
    train_df,
    test_df
):

    print("\n" + "#" * 70)
    print("MODEL 2 OF 4: BANKING INTENT MODEL")
    print("#" * 70)

    print(
        "\nPreparing Banking77 dataset..."
    )

    print(
        f"Training records available: {len(train_df)}"
    )

    print(
        f"Testing records available:  {len(test_df)}"
    )

    print(
        "\nAvailable columns:"
    )

    print(
        list(train_df.columns)
    )

    print(
        "\nExtracting training text..."
    )

    X_train = train_df[
        "text"
    ]

    print(
        "Extracting training labels..."
    )

    y_train = train_df[
        "label"
    ]

    print(
        "\nExtracting testing text..."
    )

    X_test = test_df[
        "text"
    ]

    print(
        "Extracting testing labels..."
    )

    y_test = test_df[
        "label"
    ]

    print(
        "\nBanking intent data prepared successfully."
    )

    print(
        f"Number of unique banking intents: "
        f"{y_train.nunique()}"
    )

    train_and_save_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "banking_intent_model"
    )

    print(
        "\nBANKING INTENT MODEL COMPLETED."
    )


# ============================================================
# 3. TRAIN ETHICAL OFFER MODEL
# ============================================================

def train_ethical_model(
    custom_df
):

    print("\n" + "#" * 70)
    print("MODEL 3 OF 4: ETHICAL OFFER MODEL")
    print("#" * 70)

    print(
        "\nPreparing custom banking offer dataset..."
    )

    print(
        f"Total custom offer records: "
        f"{len(custom_df)}"
    )

    print(
        "\nAvailable columns:"
    )

    print(
        list(custom_df.columns)
    )

    print(
        "\nExtracting offer text..."
    )

    X = custom_df[
        "text"
    ]

    print(
        "Extracting ethical labels..."
    )

    y = custom_df[
        "ethical_label"
    ]

    print(
        "\nEthical label distribution:"
    )

    print(
        y.value_counts()
    )

    print(
        "\nSplitting data into training and testing sets..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print(
        "Dataset split completed."
    )

    print(
        f"Training data: {len(X_train)} records"
    )

    print(
        f"Testing data:  {len(X_test)} records"
    )

    train_and_save_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "ethical_offer_model"
    )

    print(
        "\nETHICAL OFFER MODEL COMPLETED."
    )


# ============================================================
# 4. TRAIN RISK LEVEL MODEL
# ============================================================

def train_risk_model(
    custom_df
):

    print("\n" + "#" * 70)
    print("MODEL 4 OF 4: OFFER RISK MODEL")
    print("#" * 70)

    print(
        "\nPreparing risk level dataset..."
    )

    print(
        f"Total records: {len(custom_df)}"
    )

    print(
        "\nExtracting offer text..."
    )

    X = custom_df[
        "text"
    ]

    print(
        "Extracting risk level labels..."
    )

    y = custom_df[
        "risk_level"
    ]

    print(
        "\nRisk level distribution:"
    )

    print(
        y.value_counts()
    )

    print(
        "\nSplitting data into training and testing sets..."
    )

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    print(
        "Dataset split completed."
    )

    print(
        f"Training data: {len(X_train)} records"
    )

    print(
        f"Testing data:  {len(X_test)} records"
    )

    train_and_save_model(
        X_train,
        X_test,
        y_train,
        y_test,
        "offer_risk_model"
    )

    print(
        "\nOFFER RISK MODEL COMPLETED."
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def train_all_models():

    print("\n" + "=" * 70)
    print("[5/10] STARTING DATASET PREPARATION")
    print("=" * 70)

    print(
        "\nCalling prepare_offer_scan_dataset()..."
    )

    print(
        "This will load and preprocess all datasets."
    )

    data = prepare_offer_scan_dataset()

    print(
        "\nDataset preparation completed successfully!"
    )

    print(
        "\nAvailable datasets returned:"
    )

    print(
        list(data.keys())
    )

    print(
        "\nExtracting Financial PhraseBank dataset..."
    )

    phrasebank_df = data[
        "phrasebank"
    ]

    print(
        f"Financial PhraseBank loaded: "
        f"{len(phrasebank_df)} records"
    )

    print(
        "\nExtracting Banking77 training dataset..."
    )

    banking77_train = data[
        "banking77_train"
    ]

    print(
        f"Banking77 training data loaded: "
        f"{len(banking77_train)} records"
    )

    print(
        "\nExtracting Banking77 testing dataset..."
    )

    banking77_test = data[
        "banking77_test"
    ]

    print(
        f"Banking77 testing data loaded: "
        f"{len(banking77_test)} records"
    )

    print(
        "\nExtracting custom banking offers..."
    )

    custom_df = data[
        "custom"
    ]

    print(
        f"Custom banking offers loaded: "
        f"{len(custom_df)} records"
    )

    print("\n" + "=" * 70)
    print("[6/10] DATASETS READY")
    print("=" * 70)

    print(
        "\nProceeding to Financial Sentiment Model..."
    )

    train_sentiment_model(
        phrasebank_df
    )

    print("\n" + "=" * 70)
    print("[7/10] MODEL 1 COMPLETED")
    print("=" * 70)

    print(
        "\nProceeding to Banking Intent Model..."
    )

    train_banking_intent_model(
        banking77_train,
        banking77_test
    )

    print("\n" + "=" * 70)
    print("[8/10] MODEL 2 COMPLETED")
    print("=" * 70)

    print(
        "\nProceeding to Ethical Offer Model..."
    )

    train_ethical_model(
        custom_df
    )

    print("\n" + "=" * 70)
    print("[9/10] MODEL 3 COMPLETED")
    print("=" * 70)

    print(
        "\nProceeding to Offer Risk Model..."
    )

    train_risk_model(
        custom_df
    )

    print("\n" + "=" * 70)
    print("[10/10] ALL TRAINING TASKS COMPLETED")
    print("=" * 70)

    print(
        "\nSUCCESS: ALL ETHOSBANK AI OFFER SCAN MODELS "
        "TRAINED SUCCESSFULLY!"
    )

    print(
        "\nSaved models:"
    )

    for file_name in os.listdir(MODEL_DIR):

        print(
            f"    ✓ {file_name}"
        )

    print("\nModel location:")

    print(
        MODEL_DIR
    )

    print("\n" + "=" * 70)
    print("ETHOSBANK AI TRAINING PROGRAM FINISHED")
    print("=" * 70 + "\n")


# ============================================================
# RUN TRAINING
# ============================================================

if __name__ == "__main__":

    print(
        "\nMain program execution started."
    )

    train_all_models()

    print(
        "Python program execution finished successfully."
    )