import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

def train_models():
    print("Starting ML Model Training pipeline...")
    os.makedirs('analytics_engine/saved_models', exist_ok=True)
    # 1. Train Models using cs-training.csv (Wellness Score & Debt Risk)

    csv_path = 'analytics_engine/cs-training.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    print("Loading cs-training.csv...")
    df_cs = pd.read_csv(csv_path)

    # Preprocessing
    # Impute missing values
    income_median = df_cs['MonthlyIncome'].median()
    dependents_median = df_cs['NumberOfDependents'].median()
    df_cs['MonthlyIncome'] = df_cs['MonthlyIncome'].fillna(income_median)
    df_cs['NumberOfDependents'] = df_cs['NumberOfDependents'].fillna(dependents_median)

    # Save imputation values to load them in prediction service
    joblib.dump({
        'MonthlyIncome': income_median,
        'NumberOfDependents': dependents_median
    }, 'analytics_engine/saved_models/cs_imputations.pkl')

    # Features and target
    X_cs = df_cs.drop(columns=['Unnamed: 0', 'SeriousDlqin2yrs'])
    y_cs = df_cs['SeriousDlqin2yrs']

    # 1A. Financial Wellness Score - Random Forest
    print("Training Random Forest Classifier for Wellness Score...")
    rf_wellness = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rf_wellness.fit(X_cs, y_cs)
    joblib.dump(rf_wellness, 'analytics_engine/saved_models/wellness_rf.pkl')
    print("Saved wellness_rf.pkl")

    # 1B. Debt Risk - Random Forest & XGBoost
    print("Training Random Forest Classifier for Debt Risk...")
    rf_debt = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
    rf_debt.fit(X_cs, y_cs)
    joblib.dump(rf_debt, 'analytics_engine/saved_models/debt_rf.pkl')
    print("Saved debt_rf.pkl")

    print("Training XGBoost Classifier for Debt Risk...")
    # XGBoost requires features to be numeric and no spaces/symbols in column names if using booster
    xgb_debt = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1)
    xgb_debt.fit(X_cs, y_cs)
    joblib.dump(xgb_debt, 'analytics_engine/saved_models/debt_xgb.pkl')
    print("Saved debt_xgb.pkl")

 
    xlsx_path = 'analytics_engine/personal_transactions.xlsx'
    if not os.path.exists(xlsx_path):
        print(f"Error: {xlsx_path} not found.")
        return

    print("Loading personal_transactions.xlsx...")
    df_tx = pd.read_excel(xlsx_path)

    # Let's clean amounts and transaction types
    df_tx['Amount'] = pd.to_numeric(df_tx['Amount'], errors='coerce').fillna(0.0)
    df_tx['Transaction Type'] = df_tx['Transaction Type'].str.strip().str.lower()
    df_tx['Category'] = df_tx['Category'].str.strip()

    # 2A. Savings Forecast - Linear Regression
    print("Processing savings time series for Linear Regression...")
    # Group by YYYY-MM
    monthly_tx = []
    for month, group in df_tx.groupby('Month'):
        credits = group[group['Transaction Type'] == 'credit']['Amount'].sum()
        debits = group[group['Transaction Type'] == 'debit']['Amount'].sum()
        net_savings = credits - debits
        monthly_tx.append({
            'Month': month,
            'Credits': credits,
            'Debits': debits,
            'NetSavings': net_savings
        })
    
    df_monthly = pd.DataFrame(monthly_tx).sort_values('Month').reset_index(drop=True)
    df_monthly['CumulativeSavings'] = df_monthly['NetSavings'].cumsum()

    # Fit Linear Regression: X = Month Index (0, 1, 2...), y = CumulativeSavings
    X_savings = np.arange(len(df_monthly)).reshape(-1, 1)
    y_savings = df_monthly['CumulativeSavings'].values

    lr_savings = LinearRegression()
    lr_savings.fit(X_savings, y_savings)
    joblib.dump(lr_savings, 'analytics_engine/saved_models/savings_lr.pkl')
    print("Saved savings_lr.pkl")

    # Save baseline stats for scaling savings projection for new users
    avg_monthly_net_savings = df_monthly['NetSavings'].mean()
    avg_monthly_income = df_monthly['Credits'].mean()
    joblib.dump({
        'avg_monthly_net_savings': avg_monthly_net_savings,
        'avg_monthly_income': avg_monthly_income,
        'total_months_dataset': len(df_monthly)
    }, 'analytics_engine/saved_models/savings_baselines.pkl')
    print("Saved savings_baselines.pkl")

    # 2B. Spending Pattern Detection - K-Means
    print("Processing spending patterns for K-Means Clustering...")
    
    # Map raw categories to standard categories
    category_mapping = {
        'Groceries': 'Food', 'Food & Dining': 'Food', 'Coffee Shops': 'Food', 'Fast Food': 'Food', 'Restaurants': 'Food', 'Alcohol & Bars': 'Food',
        'Shopping': 'Shopping', 'Electronics & Software': 'Shopping',
        'Gas & Fuel': 'Travel', 'Auto Insurance': 'Travel',
        'Utilities': 'Bills', 'Internet': 'Bills', 'Mobile Phone': 'Bills', 'Television': 'Bills', 'City Water Charges': 'Bills', 'Power Company': 'Bills', 'Phone Company': 'Bills', 'Gas Company': 'Bills', 'Mortgage & Rent': 'Bills',
        'Haircut': 'Healthcare',
        'Music': 'Subscription', 'Movies & Dvds': 'Subscription', 'Netflix': 'Subscription', 'Spotify': 'Subscription', 'Entertainment': 'Subscription',
    }
    
    standard_categories = ['Food', 'Shopping', 'Travel', 'Bills', 'Healthcare', 'Education', 'Investment', 'Subscription', 'Other']

    # For K-Means, we want to cluster monthly spending distributions.
    # So we get total spending in debits for each month and each category.
    monthly_category_spend = []
    
    for month, group in df_tx.groupby('Month'):
        debits_group = group[group['Transaction Type'] == 'debit']
        total_debit = debits_group['Amount'].sum()
        
        row = {'Month': month}
        for cat in standard_categories:
            row[cat] = 0.0
            
        for _, r_tx in debits_group.iterrows():
            raw_cat = r_tx['Category']
            mapped_cat = category_mapping.get(raw_cat, 'Other')
            if mapped_cat in row:
                row[mapped_cat] += r_tx['Amount']
                
        # Normalize to percentages to capture relative distribution
        if total_debit > 0:
            for cat in standard_categories:
                row[cat] = (row[cat] / total_debit) * 100.0
                
        monthly_category_spend.append(row)
        
    df_spend = pd.DataFrame(monthly_category_spend)
    X_spend = df_spend[standard_categories].values

    # Standardize data before clustering using StandardScaler
    scaler = StandardScaler()
    X_spend_scaled = scaler.fit_transform(X_spend)

    # Fit K-Means
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_spend_scaled)

    joblib.dump(scaler, 'analytics_engine/saved_models/spending_scaler.pkl')
    joblib.dump(kmeans, 'analytics_engine/saved_models/spending_kmeans.pkl')
    print("Saved spending_scaler.pkl and spending_kmeans.pkl")

    # Let's save cluster descriptions and characteristics
    cluster_centers = kmeans.cluster_centers_
    # Inverse transform centers to see original scale percentages
    centers_original = scaler.inverse_transform(cluster_centers)
    
    cluster_profiles = {}
    for i, center in enumerate(centers_original):
        # Sort categories by percentage spend
        sorted_indices = np.argsort(center)[::-1]
        top_cats = [(standard_categories[idx], round(center[idx], 2)) for idx in sorted_indices]
        cluster_profiles[i] = {
            'profile_name': f"Pattern {i+1}",
            'top_categories': top_cats,
            'center_percentages': {cat: round(center[idx], 2) for idx, cat in enumerate(standard_categories)}
        }
        
    joblib.dump(cluster_profiles, 'analytics_engine/saved_models/spending_clusters_metadata.pkl')
    print("Saved spending_clusters_metadata.pkl")
    print("ML model training completed successfully!")

if __name__ == '__main__':
    train_models()
