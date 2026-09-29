import os
import re
import pandas as pd
import numpy as np

from datasets import load_dataset
from sklearn.model_selection import train_test_split


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHRASEBANK_DIR = os.path.join(BASE_DIR, "datasets", "raw", "financial_phrasebank")
CUSTOM_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "custom", "custom_banking_offers.csv")


# ============================================================
# ENHANCED TEXT CLEANING FOR ACCURACY BOOST
# ============================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    
    # Expand common financial/intent contractions
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'t", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'m", " am", text)

    # Clean non-standard characters while keeping key financial symbols
    text = re.sub(r"[^a-z0-9%\$₹\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# LOAD FINANCIAL PHRASEBANK
# ============================================================

def load_financial_phrasebank():
    file_path = os.path.join(PHRASEBANK_DIR, "Sentences_AllAgree.txt")
    if not os.path.exists(file_path):
        file_path = r"D:\Semester5\EthosBankAI - 2\offer_scan\datasets\raw\financial_phrasebank\Sentences_AllAgree.txt"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Financial PhraseBank file not found:\n{file_path}")

    records = []
    with open(file_path, "r", encoding="latin-1") as file:
        for line in file:
            line = line.strip()
            if not line or "@" not in line:
                continue
            sentence, label = line.rsplit("@", 1)
            records.append({
                "text": clean_text(sentence),
                "label": label.strip().lower(),
                "source": "Financial PhraseBank"
            })

    df = pd.DataFrame(records)
    print(f"Financial PhraseBank loaded: {len(df)} records")
    return df


# ============================================================
# LOAD BANKING77
# ============================================================

def load_banking77():
    print("Loading Banking77 dataset...")
    dataset = load_dataset("PolyAI/banking77")

    train_df = pd.DataFrame(dataset["train"])[["text", "label"]].copy()
    test_df = pd.DataFrame(dataset["test"])[["text", "label"]].copy()

    label_names = dataset["train"].features["label"].names
    train_df["label"] = train_df["label"].apply(lambda x: label_names[x])
    test_df["label"] = test_df["label"].apply(lambda x: label_names[x])

    train_df["text"] = train_df["text"].apply(clean_text)
    test_df["text"] = test_df["text"].apply(clean_text)

    train_df["source"] = "Banking77"
    test_df["source"] = "Banking77"

    print(f"Banking77 training records: {len(train_df)}")
    print(f"Banking77 testing records: {len(test_df)}")
    return train_df, test_df


# ============================================================
# LOAD CUSTOM BANKING OFFER DATASET
# ============================================================

def load_custom_dataset():
    custom_path = CUSTOM_DATASET_PATH
    if not os.path.exists(custom_path):
        custom_path = r"D:\Semester5\EthosBankAI - 2\offer_scan\datasets\custom\custom_banking_offers.csv"

    df = pd.read_csv(custom_path)
    print(f"Custom banking offers loaded: {len(df)} records")

    required_columns = [
        "offer_id", "bank_name", "institution_type", "product", "offer_title",
        "offer_text", "interest_rate", "processing_fee", "loan_tenure",
        "eligibility", "cta", "contains_terms_and_conditions", "mentions_processing_fee",
        "mentions_interest_rate", "mentions_penalty", "mentions_foreclosure",
        "urgency_phrase", "scarcity_phrase", "emotional_phrase", "hidden_fee_indicator",
        "transparency_score", "manipulation_score", "risk_level", "ethical_label",
        "sentiment", "complexity_score", "primary_trigger", "secondary_trigger",
        "offer_category"
    ]

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError("Custom dataset is missing columns: " + ", ".join(missing_columns))

    df["offer_text"] = df["offer_text"].fillna("").astype(str)
    df = df[df["offer_text"].str.strip() != ""].copy()

    df["text"] = (
        df["offer_title"].fillna("").astype(str)
        + " "
        + df["offer_text"].fillna("").astype(str)
    ).apply(clean_text)

    # Convert binary/flag columns to strictly numeric (0 and 1)
    flag_columns = [
        "contains_terms_and_conditions", "mentions_processing_fee",
        "mentions_interest_rate", "mentions_penalty", "mentions_foreclosure",
        "urgency_phrase", "scarcity_phrase", "emotional_phrase", "hidden_fee_indicator"
    ]

    for col in flag_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].map({
                'true': 1, '1': 1, 'yes': 1, 'y': 1,
                'false': 0, '0': 0, 'no': 0, 'n': 0,
                'none': 0, 'nan': 0, '': 0
            }).fillna(0).astype(int)

    df["ethical_label"] = df["ethical_label"].fillna("Unknown").astype(str).str.strip()
    df["risk_level"] = df["risk_level"].fillna("Unknown").astype(str).str.strip()
    df["offer_category"] = df["offer_category"].fillna("Unknown").astype(str).str.strip()

    df.reset_index(drop=True, inplace=True)
    print(f"Custom banking offers after cleaning: {len(df)} records")
    return df


# ============================================================
# LOAD ALL DATA
# ============================================================

def load_all_datasets():
    phrasebank_df = load_financial_phrasebank()
    banking77_train, banking77_test = load_banking77()
    custom_df = load_custom_dataset()
    return phrasebank_df, banking77_train, banking77_test, custom_df


def prepare_offer_scan_dataset():
    phrasebank_df, banking77_train, banking77_test, custom_df = load_all_datasets()

    print("\nDataset Summary")
    print("-------------------------")
    print("Financial PhraseBank:", len(phrasebank_df))
    print("Banking77:", len(banking77_train))
    print("Custom offers:", len(custom_df))

    return {
        "phrasebank": phrasebank_df,
        "banking77_train": banking77_train,
        "banking77_test": banking77_test,
        "custom": custom_df
    }


if __name__ == "__main__":
    data = prepare_offer_scan_dataset()
    print("\nPreprocessing completed successfully.")