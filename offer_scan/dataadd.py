import os
import random
import pandas as pd

# 1. Resolve CSV location
CSV_PATH = r"D:\Semester5\EthosBankAI - 2\offer_scan\datasets\custom\custom_banking_offers.csv"
if not os.path.exists(CSV_PATH):
    CSV_PATH = "custom_banking_offers.csv"

df_existing = pd.read_csv(CSV_PATH)
start_idx = len(df_existing) + 1
end_idx = start_idx + 2000

# 2. Setup lookup entities & balanced templates
bank_names = [
    "Apex Microfinance", "FastCredit Digital", "SmartWealth Lenders", "Global Consumer Finance",
    "FlexiPay Capital", "Urban Micro Credit", "Direct Cash Express", "Prime Auto Finance",
    "TrustPay Gold", "QuickFund Micro", "EzLoan App", "CrediQuick Pvt Ltd", "RupeeFast Finance",
    "InstantRupee App", "LoanBazaar NBFC", "ZippyCash Lenders", "PayDay365", "MicroPay Solutions"
]

products = ["Personal Loan", "Payday Loan", "Credit Card", "BNPL", "Gold Loan", "Auto Loan"]

templates = [
    # Fraudulent
    {
        "ethical_label": "Fraudulent", "risk_level": "Critical", "sentiment": "Negative",
        "title_fmt": "Instant 0% Interest Cash Approval #{i}",
        "text_fmt": "Guaranteed instant cash with 0% interest rate! Send upfront processing fee of Rs {fee} to activate wallet.",
        "ir": "0%", "pf": "20.0%", "tenure": "3 months", "elig": "Everyone approved", "cta": "Pay Fee Now",
        "ttc": "FALSE", "mpf": True, "mir": True, "mp": False, "mfc": False,
        "urg": "Act within 10 mins", "scar": "First 50 users only", "emot": "Solve money stress now",
        "hfi": "TRUE", "trans": 12, "manip": 95, "cx": 85, "trig1": "Urgency", "trig2": "Greed", "cat": "Cashback offer"
    },
    {
        "ethical_label": "Fraudulent", "risk_level": "Critical", "sentiment": "Negative",
        "title_fmt": "Claim Rs {amt} Emergency Cash Today",
        "text_fmt": "Transfer money to account instantly! Zero documentation. Penalty fees applied after 7 days.",
        "ir": "42%", "pf": "15.0%", "tenure": "1 month", "elig": "No credit check required", "cta": "Claim Money Now",
        "ttc": "FALSE", "mpf": False, "mir": False, "mp": True, "mfc": False,
        "urg": "Offer expires today", "scar": "Limited slots left", "emot": "Emergency safety net",
        "hfi": "TRUE", "trans": 18, "manip": 90, "cx": 80, "trig1": "Fear", "trig2": "Urgency", "cat": "Personal loan offer"
    },
    # Deceptive
    {
        "ethical_label": "Deceptive", "risk_level": "High", "sentiment": "Neutral",
        "title_fmt": "Lifetime Free Platinum Credit Card #{i}",
        "text_fmt": "Enjoy lifetime free card with zero annual charges! *Fee waived only on annual spends above Rs {amt}.",
        "ir": "38%", "pf": "3.5%", "tenure": "12 months", "elig": "Salary above 25k", "cta": "Apply Now",
        "ttc": "TRUE", "mpf": True, "mir": False, "mp": True, "mfc": False,
        "urg": "Apply this week", "scar": "Exclusive invite", "emot": "Upgrade your lifestyle",
        "hfi": "TRUE", "trans": 35, "manip": 78, "cx": 68, "trig1": "Status", "trig2": "Savings", "cat": "Credit card offer"
    },
    {
        "ethical_label": "Deceptive", "risk_level": "High", "sentiment": "Neutral",
        "title_fmt": "Lowest 5.99% Fixed Rate Home Loan #{i}",
        "text_fmt": "Get fixed home loan rates starting at 5.99% for initial 6 months. Rate jumps to 12.5% from 7th month.",
        "ir": "12.5%", "pf": "1.5%", "tenure": "20 years", "elig": "Home buyers", "cta": "Check Eligibility",
        "ttc": "TRUE", "mpf": True, "mir": True, "mp": True, "mfc": True,
        "urg": "Limited period deal", "scar": "Special rate quota", "emot": "Build your dream house",
        "hfi": "TRUE", "trans": 42, "manip": 72, "cx": 75, "trig1": "Savings", "trig2": "Security", "cat": "Home loan offer"
    },
    # Moderately Manipulative
    {
        "ethical_label": "Moderately Manipulative", "risk_level": "Moderate", "sentiment": "Positive",
        "title_fmt": "Shop Now Pay Later with 0% Interest EMI #{i}",
        "text_fmt": "Zero cost EMI on all purchases! Late penalty fee of Rs {fee} per day plus merchant convenience charges apply.",
        "ir": "0%", "pf": "4.0%", "tenure": "6 months", "elig": "BNPL eligible", "cta": "Activate BNPL",
        "ttc": "TRUE", "mpf": True, "mir": True, "mp": True, "mfc": False,
        "urg": "Offer ends tonight", "scar": "Few coupons left", "emot": "Shop without waiting",
        "hfi": "TRUE", "trans": 48, "manip": 65, "cx": 55, "trig1": "Convenience", "trig2": "Urgency", "cat": "Reward point offer"
    },
    {
        "ethical_label": "Moderately Manipulative", "risk_level": "Moderate", "sentiment": "Positive",
        "title_fmt": "Pre-Approved Personal Loan Up To {amt} Lakhs",
        "text_fmt": "Pre-approved loan disbursed in 2 hours! Loan subject to upfront processing charges and mandatory insurance.",
        "ir": "18.5%", "pf": "3.0%", "tenure": "36 months", "elig": "Pre-screened customers", "cta": "Claim Loan",
        "ttc": "TRUE", "mpf": True, "mir": True, "mp": True, "mfc": True,
        "urg": "Valid for 24 hours", "scar": "Reserved for you", "emot": "Fulfill your dream vacation",
        "hfi": "FALSE", "trans": 52, "manip": 58, "cx": 50, "trig1": "Ease", "trig2": "Aspiration", "cat": "Personal loan offer"
    }
]

# 3. Generate 2,000 new rows using class balancing and continuous feature jitter
random.seed(4040)
fraud_tmpls = [t for t in templates if t["ethical_label"] == "Fraudulent"]
decep_tmpls = [t for t in templates if t["ethical_label"] == "Deceptive"]
mod_tmpls = [t for t in templates if t["ethical_label"] == "Moderately Manipulative"]

new_rows = []
for i in range(start_idx, end_idx):
    # Equal 1:1:1 class assignment
    cls_idx = i % 3
    tmpl = fraud_tmpls[i % len(fraud_tmpls)] if cls_idx == 0 else (decep_tmpls[i % len(decep_tmpls)] if cls_idx == 1 else mod_tmpls[i % len(mod_tmpls)])

    bank = random.choice(bank_names)
    prod = random.choice(products)
    amt_val = random.choice([25000, 50000, 75000, 100000, 250000, 500000])
    fee_val = random.choice([999, 1499, 1999, 2499, 2999, 3499])
    
    title = tmpl["title_fmt"].format(i=i, amt=amt_val, fee=fee_val)
    text = tmpl["text_fmt"].format(i=i, amt=amt_val, fee=fee_val)
    
    # Feature jittering to prevent exact parameter memorization (overfitting defense)
    manip_score = min(100, max(0, int(tmpl["manip"] + random.randint(-5, 5))))
    trans_score = min(100, max(0, int(tmpl["trans"] + random.randint(-5, 5))))
    cx_score = min(100, max(0, int(tmpl["cx"] + random.randint(-5, 5))))
    
    row = {
        "offer_id": f"OFFER_{i}",
        "bank_name": bank,
        "institution_type": "NBFC" if "NBFC" in bank else ("Fintech App" if "App" in bank else "Bank"),
        "product": prod,
        "offer_title": title,
        "offer_text": text,
        "interest_rate": tmpl["ir"],
        "processing_fee": tmpl["pf"],
        "loan_tenure": tmpl["tenure"],
        "eligibility": tmpl["elig"],
        "cta": tmpl["cta"],
        "contains_terms_and_conditions": tmpl["ttc"],
        "mentions_processing_fee": tmpl["mpf"],
        "mentions_interest_rate": tmpl["mir"],
        "mentions_penalty": tmpl["mp"],
        "mentions_foreclosure": tmpl["mfc"],
        "urgency_phrase": tmpl["urg"],
        "scarcity_phrase": tmpl["scar"],
        "emotional_phrase": tmpl["emot"],
        "hidden_fee_indicator": tmpl["hfi"],
        "transparency_score": trans_score,
        "manipulation_score": manip_score,
        "risk_level": tmpl["risk_level"],
        "ethical_label": tmpl["ethical_label"],
        "sentiment": tmpl["sentiment"],
        "complexity_score": cx_score,
        "primary_trigger": tmpl["trig1"],
        "secondary_trigger": tmpl["trig2"],
        "offer_category": tmpl["cat"]
    }
    new_rows.append(row)

# 4. Append and save
df_2000 = pd.DataFrame(new_rows)
df_final = pd.concat([df_existing, df_2000], ignore_index=True)
df_final.to_csv(CSV_PATH, index=False)

print(f"Dataset successfully updated to {len(df_final)} records!")