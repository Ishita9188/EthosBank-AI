
# Create your tests here.
import os
import joblib
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "analytics_engine/saved_models"

CS_DATASET = "analytics_engine/cs-training.csv"
TRANSACTION_DATASET = "analytics_engine/personal_transactions.xlsx"


# ============================================================
# HELPER FUNCTION
# ============================================================

def load_model(filename):
    """
    Load a stored machine learning model.
    """
    path = os.path.join(MODEL_DIR, filename)

    if not os.path.exists(path):
        print(f"[ERROR] Model not found: {path}")
        return None

    try:
        model = joblib.load(path)
        print(f"[OK] Loaded: {filename}")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load {filename}: {e}")
        return None


# ============================================================
# 1. LOAD CS-TRAINING DATA
# ============================================================

def load_cs_dataset():

    print("\n" + "=" * 70)
    print("LOADING CS-TRAINING DATASET")
    print("=" * 70)

    if not os.path.exists(CS_DATASET):
        print(f"[ERROR] Dataset not found: {CS_DATASET}")
        return None, None

    df = pd.read_csv(CS_DATASET)

    print(f"Dataset shape: {df.shape}")

    # Same preprocessing used during training
    income_median = df["MonthlyIncome"].median()
    dependents_median = df["NumberOfDependents"].median()

    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(income_median)
    df["NumberOfDependents"] = (
        df["NumberOfDependents"].fillna(dependents_median)
    )

    # Features and target
    X = df.drop(
        columns=["Unnamed: 0", "SeriousDlqin2yrs"],
        errors="ignore"
    )

    y = df["SeriousDlqin2yrs"]

    print(f"Features: {X.shape[1]}")
    print(f"Samples: {X.shape[0]}")
    print(f"Target classes: {sorted(y.unique())}")

    return X, y


# ============================================================
# 2. CLASSIFICATION MODEL EVALUATION
# ============================================================

def evaluate_classifier(model, model_name, X, y):

    print("\n" + "=" * 70)
    print(f"{model_name.upper()} — CLASSIFICATION EVALUATION")
    print("=" * 70)

    try:
        predictions = model.predict(X)

        accuracy = accuracy_score(y, predictions)

        precision = precision_score(
            y,
            predictions,
            average="binary",
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            average="binary",
            zero_division=0
        )

        f1 = f1_score(
            y,
            predictions,
            average="binary",
            zero_division=0
        )

        cm = confusion_matrix(y, predictions)

        print(f"\nAccuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"Precision : {precision:.4f} ({precision * 100:.2f}%)")
        print(f"Recall    : {recall:.4f} ({recall * 100:.2f}%)")
        print(f"F1 Score  : {f1:.4f} ({f1 * 100:.2f}%)")

        print("\nConfusion Matrix:")
        print(cm)

        print("\nClassification Report:")
        print(
            classification_report(
                y,
                predictions,
                zero_division=0
            )
        )

        # ROC-AUC if probability prediction is available
        if hasattr(model, "predict_proba"):

            try:
                from sklearn.metrics import roc_auc_score

                probabilities = model.predict_proba(X)[:, 1]

                auc = roc_auc_score(y, probabilities)

                print(
                    f"ROC-AUC   : {auc:.4f} ({auc * 100:.2f}%)"
                )

            except Exception as e:
                print(f"ROC-AUC could not be calculated: {e}")

        return {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }

    except Exception as e:

        print(f"[ERROR] Evaluation failed for {model_name}: {e}")

        return None


# ============================================================
# 3. SAVINGS FORECAST — LINEAR REGRESSION
# ============================================================

def evaluate_savings_model():

    print("\n" + "=" * 70)
    print("SAVINGS FORECAST — LINEAR REGRESSION")
    print("=" * 70)

    if not os.path.exists(TRANSACTION_DATASET):
        print(
            f"[ERROR] Dataset not found: "
            f"{TRANSACTION_DATASET}"
        )
        return None

    model = load_model("savings_lr.pkl")

    if model is None:
        return None

    # Load transaction data
    df = pd.read_excel(TRANSACTION_DATASET)

    # Same cleaning used during training
    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    ).fillna(0.0)

    df["Transaction Type"] = (
        df["Transaction Type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Monthly aggregation
    monthly_tx = []

    for month, group in df.groupby("Month"):

        credits = group[
            group["Transaction Type"] == "credit"
        ]["Amount"].sum()

        debits = group[
            group["Transaction Type"] == "debit"
        ]["Amount"].sum()

        net_savings = credits - debits

        monthly_tx.append({
            "Month": month,
            "Credits": credits,
            "Debits": debits,
            "NetSavings": net_savings
        })

    df_monthly = (
        pd.DataFrame(monthly_tx)
        .sort_values("Month")
        .reset_index(drop=True)
    )

    df_monthly["CumulativeSavings"] = (
        df_monthly["NetSavings"].cumsum()
    )

    # Same X and y used during training
    X = np.arange(
        len(df_monthly)
    ).reshape(-1, 1)

    y = df_monthly["CumulativeSavings"].values

    # Predictions from stored model
    predictions = model.predict(X)

    # Regression metrics
    mae = mean_absolute_error(y, predictions)

    mse = mean_squared_error(y, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y, predictions)

    print(f"\nMAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # Show actual vs predicted
    results = pd.DataFrame({
        "Month": df_monthly["Month"],
        "Actual Savings": y,
        "Predicted Savings": predictions
    })

    print("\nActual vs Predicted Savings:")
    print(results.to_string(index=False))

    return {
        "Model": "Savings Linear Regression",
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# 4. K-MEANS SPENDING BEHAVIOUR EVALUATION
# ============================================================

def evaluate_spending_model():

    print("\n" + "=" * 70)
    print("SPENDING BEHAVIOUR — K-MEANS CLUSTERING")
    print("=" * 70)

    if not os.path.exists(TRANSACTION_DATASET):
        print(
            f"[ERROR] Dataset not found: "
            f"{TRANSACTION_DATASET}"
        )
        return None

    scaler = load_model("spending_scaler.pkl")
    kmeans = load_model("spending_kmeans.pkl")

    if scaler is None or kmeans is None:
        return None

    df = pd.read_excel(TRANSACTION_DATASET)

    # Same cleaning used during training
    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    ).fillna(0.0)

    df["Transaction Type"] = (
        df["Transaction Type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    df["Category"] = (
        df["Category"]
        .astype(str)
        .str.strip()
    )

    # Category mapping from training file
    category_mapping = {

        "Groceries": "Food",
        "Food & Dining": "Food",
        "Coffee Shops": "Food",
        "Fast Food": "Food",
        "Restaurants": "Food",
        "Alcohol & Bars": "Food",

        "Shopping": "Shopping",
        "Electronics & Software": "Shopping",

        "Gas & Fuel": "Travel",
        "Auto Insurance": "Travel",

        "Utilities": "Bills",
        "Internet": "Bills",
        "Mobile Phone": "Bills",
        "Television": "Bills",
        "City Water Charges": "Bills",
        "Power Company": "Bills",
        "Phone Company": "Bills",
        "Gas Company": "Bills",
        "Mortgage & Rent": "Bills",

        "Haircut": "Healthcare",

        "Music": "Subscription",
        "Movies & Dvds": "Subscription",
        "Netflix": "Subscription",
        "Spotify": "Subscription",
        "Entertainment": "Subscription"
    }

    standard_categories = [
        "Food",
        "Shopping",
        "Travel",
        "Bills",
        "Healthcare",
        "Education",
        "Investment",
        "Subscription",
        "Other"
    ]

    # Build monthly category distributions
    monthly_category_spend = []

    for month, group in df.groupby("Month"):

        debits_group = group[
            group["Transaction Type"] == "debit"
        ]

        total_debit = debits_group["Amount"].sum()

        row = {"Month": month}

        for category in standard_categories:
            row[category] = 0.0

        for _, transaction in debits_group.iterrows():

            raw_category = transaction["Category"]

            mapped_category = category_mapping.get(
                raw_category,
                "Other"
            )

            if mapped_category in row:

                row[mapped_category] += (
                    transaction["Amount"]
                )

        # Convert to percentage distribution
        if total_debit > 0:

            for category in standard_categories:

                row[category] = (
                    row[category] /
                    total_debit
                ) * 100.0

        monthly_category_spend.append(row)

    df_spend = pd.DataFrame(
        monthly_category_spend
    )

    X = df_spend[
        standard_categories
    ].values

    # Apply STORED scaler
    X_scaled = scaler.transform(X)

    # Apply STORED K-Means model
    clusters = kmeans.predict(X_scaled)

    # Number of clusters
    print(
        f"\nNumber of clusters: "
        f"{kmeans.n_clusters}"
    )

    # Cluster distribution
    unique, counts = np.unique(
        clusters,
        return_counts=True
    )

    print("\nCluster Distribution:")

    for cluster, count in zip(unique, counts):

        percentage = (
            count / len(clusters)
        ) * 100

        print(
            f"Cluster {cluster + 1}: "
            f"{count} months "
            f"({percentage:.2f}%)"
        )

    # Silhouette score
    if len(set(clusters)) > 1:

        silhouette = silhouette_score(
            X_scaled,
            clusters
        )

        print(
            f"\nSilhouette Score: "
            f"{silhouette:.4f}"
        )

    else:

        silhouette = None

        print(
            "\nSilhouette Score cannot "
            "be calculated because only "
            "one cluster is present."
        )

    return {
        "Model": "Spending K-Means",
        "Clusters": kmeans.n_clusters,
        "Silhouette Score": silhouette
    }


# ============================================================
# 5. VERIFY ADDITIONAL STORED FILES
# ============================================================

def verify_supporting_models():

    print("\n" + "=" * 70)
    print("VERIFYING STORED MODEL FILES")
    print("=" * 70)

    files = [
        "cs_imputations.pkl",
        "wellness_rf.pkl",
        "debt_rf.pkl",
        "debt_xgb.pkl",
        "savings_lr.pkl",
        "savings_baselines.pkl",
        "spending_scaler.pkl",
        "spending_kmeans.pkl",
        "spending_clusters_metadata.pkl"
    ]

    loaded = 0

    for filename in files:

        path = os.path.join(
            MODEL_DIR,
            filename
        )

        if os.path.exists(path):

            try:

                joblib.load(path)

                print(
                    f"[OK] {filename}"
                )

                loaded += 1

            except Exception as e:

                print(
                    f"[ERROR] {filename}: {e}"
                )

        else:

            print(
                f"[MISSING] {filename}"
            )

    print(
        f"\nSuccessfully loaded "
        f"{loaded}/{len(files)} stored files."
    )


# ============================================================
# 6. MAIN TESTING PIPELINE
# ============================================================

def test_all_models():

    print("\n")
    print("=" * 70)
    print("ETHOSBANK AI — ML MODEL TESTING PIPELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Verify all saved files
    # --------------------------------------------------------

    verify_supporting_models()

    # --------------------------------------------------------
    # Load classification dataset
    # --------------------------------------------------------

    X_cs, y_cs = load_cs_dataset()

    classification_results = []

    if X_cs is not None and y_cs is not None:

        # ----------------------------------------------------
        # Wellness Random Forest
        # ----------------------------------------------------

        wellness_model = load_model(
            "wellness_rf.pkl"
        )

        if wellness_model is not None:

            result = evaluate_classifier(
                wellness_model,
                "Financial Wellness - Random Forest",
                X_cs,
                y_cs
            )

            if result:
                classification_results.append(result)

        # ----------------------------------------------------
        # Debt Risk Random Forest
        # ----------------------------------------------------

        debt_rf_model = load_model(
            "debt_rf.pkl"
        )

        if debt_rf_model is not None:

            result = evaluate_classifier(
                debt_rf_model,
                "Debt Risk - Random Forest",
                X_cs,
                y_cs
            )

            if result:
                classification_results.append(result)

        # ----------------------------------------------------
        # Debt Risk XGBoost
        # ----------------------------------------------------

        debt_xgb_model = load_model(
            "debt_xgb.pkl"
        )

        if debt_xgb_model is not None:

            result = evaluate_classifier(
                debt_xgb_model,
                "Debt Risk - XGBoost",
                X_cs,
                y_cs
            )

            if result:
                classification_results.append(result)

    # --------------------------------------------------------
    # Savings Forecast
    # --------------------------------------------------------

    savings_result = evaluate_savings_model()

    # --------------------------------------------------------
    # Spending Behaviour
    # --------------------------------------------------------

    spending_result = evaluate_spending_model()

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 70)

    # Classification summary
    if classification_results:

        print("\nCLASSIFICATION MODELS")
        print("-" * 70)

        results_df = pd.DataFrame(
            classification_results
        )

        print(
            results_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}"
            )
        )

    # Regression summary
    if savings_result:

        print("\n\nREGRESSION MODEL")
        print("-" * 70)

        print(
            f"Model : {savings_result['Model']}"
        )

        print(
            f"MAE   : "
            f"{savings_result['MAE']:.4f}"
        )

        print(
            f"MSE   : "
            f"{savings_result['MSE']:.4f}"
        )

        print(
            f"RMSE  : "
            f"{savings_result['RMSE']:.4f}"
        )

        print(
            f"R²    : "
            f"{savings_result['R2']:.4f}"
        )

    # Clustering summary
    if spending_result:

        print("\n\nCLUSTERING MODEL")
        print("-" * 70)

        print(
            f"Model            : "
            f"{spending_result['Model']}"
        )

        print(
            f"Number of Clusters: "
            f"{spending_result['Clusters']}"
        )

        if spending_result["Silhouette Score"] is not None:

            print(
                f"Silhouette Score : "
                f"{spending_result['Silhouette Score']:.4f}"
            )

    print("\n" + "=" * 70)
    print("MODEL TESTING COMPLETED")
    print("=" * 70)


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    test_all_models()