import os
import re
import joblib
import pandas as pd

# ============================================================
# ETHOSBANK AI - HYBRID OFFER SCAN V3
# ============================================================

print("\n" + "=" * 75)
print("ETHOSBANK AI - HYBRID OFFER SCAN V3")
print("=" * 75)


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "ml_models")

print(f"\nModel directory: {MODEL_DIR}")


# ============================================================
# MODEL PATHS
# ============================================================

MODEL_PATHS = {
    "sentiment": os.path.join(MODEL_DIR, "financial_sentiment_model_v2.pkl"),
    "intent": os.path.join(MODEL_DIR, "banking_intent_model_v2.pkl"),
    "ethical": os.path.join(MODEL_DIR, "ethical_offer_model_v2.pkl"),
    "risk": os.path.join(MODEL_DIR, "offer_risk_model_v2.pkl"),
}


# ============================================================
# DISTILBART / NLP CONFIGURATION
# ============================================================

DISTILBART_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
summarizer = None


def load_distilbart():
    global summarizer
    print("\n[AI NLP] Loading DistilBART...")
    try:
        from transformers import pipeline
        summarizer = pipeline("text-generation", model=DISTILBART_MODEL_NAME)
        print("[SUCCESS] DistilBART loaded successfully.")
    except Exception as e:
        print("[WARNING] DistilBART could not be loaded.")
        print(f"[WARNING] {e}")
        summarizer = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_name):
    print(f"\n[MODEL LOAD] Loading {model_name} model...")
    model_path = MODEL_PATHS.get(model_name)
    if not model_path or not os.path.exists(model_path):
        print(f"[ERROR] Model file not found: {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        print(f"[SUCCESS] {model_name} model loaded.")
        return model
    except Exception as e:
        print(f"[ERROR] Could not load {model_name} model: {e}")
        return None


# ============================================================
# LOAD ALL TRAINED MODELS & PRETRAINED NLP
# ============================================================

print("\n[1/8] Loading trained V2 models...")
sentiment_model = load_model("sentiment")
intent_model = load_model("intent")
ethical_model = load_model("ethical")
risk_model = load_model("risk")
print("\n[1/8] V2 model loading completed.")

load_distilbart()


# ============================================================
# SAFE MODEL PREDICTION HELPERS
# ============================================================

def safe_predict_sentiment(cleaned_text):
    print("\n[MODEL 1] Running Financial Sentiment V2...")
    if not sentiment_model:
        return "Unknown"
    try:
        input_df = pd.DataFrame({"text": [cleaned_text]})
        prediction = sentiment_model.predict(input_df)[0]
        print(f"[RESULT] Sentiment = {prediction}")
        return str(prediction)
    except Exception as e:
        print(f"[WARNING] Sentiment prediction failed: {e}")
        return "neutral"


def safe_predict_banking_intent(cleaned_text):
    print("\n[MODEL 2] Running Banking Intent V2...")
    if not intent_model:
        return "Unknown"
    try:
        input_df = pd.DataFrame({"text": [cleaned_text]})
        prediction = intent_model.predict(input_df)[0]
        print(f"[RESULT] Banking Intent = {prediction}")
        return str(prediction)
    except Exception as e:
        print(f"[WARNING] Banking intent prediction failed: {e}")
        return "General_Information"


def safe_predict_ethical(offer_features_df, manipulation_score):
    print("\n[MODEL 3] Running Ethical Offer V2...")
    if not ethical_model:
        if manipulation_score >= 75:
            return "Highly Manipulative"
        elif manipulation_score >= 50:
            return "Moderately Manipulative"
        elif manipulation_score >= 25:
            return "Potentially Manipulative"
        else:
            return "Ethical"
    try:
        prediction = ethical_model.predict(offer_features_df)[0]
        print(f"[RESULT] Ethical Classification = {prediction}")
        return str(prediction)
    except Exception as e:
        print(f"[WARNING] Ethical model prediction failed: {e}")
        if manipulation_score >= 50:
            return "Potentially Manipulative"
        return "Ethical"


def safe_predict_ml_risk(offer_features_df):
    print("\n[MODEL 4] Running Offer Risk V2...")
    if not risk_model:
        return "Low"
    try:
        prediction = risk_model.predict(offer_features_df)[0]
        print(f"[ML RESULT] Risk Model = {prediction}")
        return str(prediction)
    except Exception as e:
        print(f"[WARNING] Offer Risk prediction failed: {e}")
        return "Low"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text_lower = text.lower()

    contractions = {
        r"\bcan't\b": "cannot",
        r"\bwon't\b": "will not",
        r"\bn't\b": " not",
        r"\b're\b": " are",
        r"\b's\b": " is",
        r"\b'd\b": " would",
        r"\b'll\b": " will",
        r"\b've\b": " have",
        r"\b'm\b": " am",
    }

    for pattern, replacement in contractions.items():
        text_lower = re.sub(pattern, replacement, text_lower)

    text_lower = re.sub(r"https?://", " URLSCHEME ", text_lower)
    text_lower = re.sub(r"[^\w\s%₹$:/.-]", " ", text_lower)
    text_lower = text_lower.replace("URLSCHEME", "https://")
    text_lower = re.sub(r"\s+", " ", text_lower).strip()

    return text_lower


# ============================================================
# DISTILBART / SUMMARY PROCESSING
# ============================================================

def distilbart_process(original_text, cleaned_text, patterns, offer_category):
    words = cleaned_text.split()

    if summarizer and len(words) >= 25:
        try:
            limited_text = " ".join(words[:500])
            summary = summarizer(limited_text, max_length=120, min_length=25, do_sample=False)
            processed = summary[0]["summary_text"].strip()
            if processed:
                return processed
        except Exception as e:
            print(f"[WARNING] DistilBART processing failed: {e}")

    summary_parts = []
    if patterns.get("is_safety_warning"):
        return "Official security warning advising users never to share credentials or financial credentials."
    if patterns.get("is_maintenance_notice"):
        return "Official bank notice regarding scheduled service maintenance."

    if patterns.get("account_threat"):
        summary_parts.append("Urgent account threat/suspension warning")
    if patterns.get("sensitive_requests"):
        info_str = ", ".join(patterns["sensitive_requests"].keys())
        summary_parts.append(f"Request for sensitive financial data ({info_str})")
    if patterns.get("refund_bait"):
        summary_parts.append("Refund/payment bait message")
    if patterns.get("prize_bait"):
        summary_parts.append("Prize or lottery reward claim")
    if patterns.get("advance_fee"):
        summary_parts.append("Upfront processing/activation fee requirement")
    if patterns.get("upi_scam"):
        summary_parts.append("UPI PIN / QR code payment trap")
    if patterns.get("crypto_scam"):
        summary_parts.append("Cryptocurrency / wallet credential request")
    if patterns.get("investment_scam"):
        summary_parts.append("High-return guaranteed investment scheme")
    if patterns.get("phishing_link"):
        summary_parts.append("Action request containing external link")

    if summary_parts:
        return f"[{offer_category}] " + "; ".join(summary_parts) + "."
    
    return f"[{offer_category}] " + (cleaned_text[:150] + "..." if len(cleaned_text) > 150 else cleaned_text)


# ============================================================
# COMPREHENSIVE FRAUD PATTERN REGEX DEFINITIONS
# ============================================================

SAFETY_WARNING_PATTERNS = [
    r"\bnever share (?:your )?(?:otp|pin|password|cvv|card details|bank details)\b",
    r"\bdo not share (?:your )?(?:otp|pin|password|cvv|card details|bank details)\b",
    r"\bbank (?:will )?never ask (?:you )?for (?:otp|pin|password|cvv|bank details)\b",
    r"\bif you suspect fraud\b",
    r"\bcontact (?:your )?bank using the official number\b",
    r"\bkeep your (?:account|pin|otp) safe\b",
]

MAINTENANCE_PATTERNS = [
    r"\bscheduled maintenance\b",
    r"\bservices will be unavailable\b",
    r"\bnetbanking services will be unavailable\b",
    r"\bsystem maintenance\b",
    r"\bplanned downtime\b",
    r"\bmaintenance activity\b",
]

ACCOUNT_THREAT_PATTERNS = [
    r"\baccount (?:will be|has been) (?:suspended|blocked|deactivated|closed|restricted|frozen|terminated)\b",
    r"\byour (?:bank )?account (?:will be|has been) (?:suspended|blocked|deactivated|closed|restricted|frozen)\b",
    r"\bavoid (?:suspension|deactivation|blocking|closure)\b",
    r"\bprevent (?:suspension|deactivation|blocking|closure)\b",
    r"\bupdate (?:your )?(?:kyc|pan|aadhaar) (?:now|immediately|today)\b",
    r"\bverify (?:your )?(?:kyc|pan|aadhaar) (?:now|immediately)\b",
    r"\baccount verification required\b",
    r"\brevalidate (?:your )?account\b",
    r"\breactivate (?:your )?account\b",
    r"\bkyc (?:expired|pending|verification)\b",
]

SENSITIVE_INFO_PATTERNS = {
    "bank account": [r"\bbank account(?: number)?\b", r"\baccount number\b", r"\bbank details\b"],
    "routing number": [r"\brouting number\b", r"\bsort code\b", r"\bifsc(?: code)?\b"],
    "card details": [r"\bdebit card\b", r"\bcredit card\b", r"\bcard details\b", r"\bcard number\b"],
    "cvv": [r"\bcvv\b", r"\bcvc\b", r"\bsecurity code\b"],
    "pin": [r"\batm pin\b", r"\bcard pin\b", r"\bsecret pin\b"],
    "upi pin": [r"\bupi pin\b"],
    "password": [r"\bpassword\b", r"\blogin credentials\b", r"\bpasscode\b"],
    "otp": [r"\botp\b", r"\bone[-\s]?time password\b", r"\bverification code\b"],
    "aadhaar": [r"\baadhaar(?: number)?\b", r"\badhar\b"],
    "pan": [r"\bpan(?: number| card)?\b"],
    "ssn": [r"\bssn\b", r"\bsocial security number\b"],
    "seed phrase": [r"\bseed phrase\b", r"\brecovery phrase\b", r"\bprivate key\b"],
}

ACTION_REQUEST_PATTERNS = [
    r"\breply with\b",
    r"\bsend (?:us |your )?\b",
    r"\bprovide (?:your )?\b",
    r"\benter (?:your )?\b",
    r"\bshare (?:your )?\b",
    r"\bsubmit (?:your )?\b",
    r"\bfill out\b",
    r"\bverify (?:using|at|via)\b",
    r"\bclick (?:here|the link|below)\b",
    r"\bscan (?:this |the )?qr\b",
    r"\bapprove (?:the )?request\b",
]

REFUND_PATTERNS = [
    r"\brefund of (?:₹|\$|inr)?\s*[\d,]+(?:\.\d+)?\b",
    r"\bpending refund\b",
    r"\bunclaimed refund\b",
    r"\btax refund\b",
    r"\bamazon refund\b",
    r"\bpayment reversal\b",
    r"\bcashback refund\b",
    r"\bclaim (?:your )?refund\b",
    r"\breceive (?:your )?refund\b",
    r"\brefund processing\b",
    r"\brefund\b",
]

ADVANCE_FEE_PATTERNS = [
    r"\bpay (?:a |the |₹|\$|inr)?\s*[\d,]*\s*(?:processing |release |activation |registration |verification |clearance |transfer |service )?fee\b",
    r"\bprocessing fee\b",
    r"\bupfront payment\b",
    r"\badvance payment\b",
    r"\bdeposit required\b",
    r"\bprocessing fee of\b",
    r"\bpay ₹?\d+ (?:before|to receive)\b",
]

LOAN_SCAM_PATTERNS = [
    r"\bguaranteed loan\b",
    r"\binstant loan guaranteed\b",
    r"\bloan without verification\b",
    r"\bloan with no documents?\b",
    r"\bguaranteed approval\b",
    r"\bbad credit accepted\b",
    r"\bpay processing fee before loan\b",
    r"\bzero interest with\b",
    r"\bloan released immediately\b",
]

INVESTMENT_SCAM_PATTERNS = [
    r"\bguaranteed returns?\b",
    r"\bguaranteed profit\b",
    r"\brisk[-\s]?free investment\b",
    r"\bdouble your money\b",
    r"\btriple your money\b",
    r"\bfixed daily income\b",
    r"\bassured returns?\b",
    r"\b100% profit\b",
    r"\bno risk investment\b",
    r"\bsecret investment opportunity\b",
    r"\bexclusive investment scheme\b",
    r"\b\d+%\s*(?:monthly|daily|weekly) returns?\b",
]

CRYPTO_SCAM_PATTERNS = [
    r"\bcrypto giveaway\b",
    r"\bdouble your bitcoin\b",
    r"\bdouble your crypto\b",
    r"\bguaranteed crypto returns\b",
    r"\bwallet verification\b",
    r"\bconnect (?:your )?wallet\b",
    r"\bcrypto airdrop\b",
    r"\bsend crypto to receive\b",
]

PRIZE_PATTERNS = [
    r"\bcongratulations!you (?:have )?won\b",
    r"\bcongratulations you (?:have )?won\b",
    r"\bcongratulations! you (?:have )?won\b",
    r"\bclaim (?:your )?prize\b",
    r"\blottery winner\b",
    r"\blucky customer\b",
    r"\bselected winner\b",
    r"\bexclusive reward\b",
    r"\bclaim reward now\b",
    r"\bwon (?:₹|\$|inr)?\s*[\d,]+\b",
]

UPI_SCAM_PATTERNS = [
    r"\benter (?:your )?upi pin to receive\b",
    r"\bscan (?:this |the )?qr (?:code )?to receive\b",
    r"\bapprove (?:collect )?request to receive\b",
    r"\bupi collect request\b",
    r"\breverse payment using pin\b",
]

URGENCY_PATTERNS = [
    r"\bact now\b",
    r"\bapply now\b",
    r"\bpay now\b",
    r"\bupdate now\b",
    r"\bverify now\b",
    r"\brespond now\b",
    r"\btoday only\b",
    r"\bexpires today\b",
    r"\bimmediately\b",
    r"\burgent\b",
    r"\bhurry\b",
    r"\blast chance\b",
    r"\bwithin \d+ hours?\b",
    r"\bwithin \d+ mins?\b",
]

SCARCITY_PATTERNS = [
    r"\blimited time\b",
    r"\blimited slots?\b",
    r"\bonly \d+ left\b",
    r"\bfew slots?\b",
    r"\bexclusive offer\b",
]

URL_PATTERNS = [
    r"https?://[^\s]+",
    r"\bwww\.[^\s]+\b",
    r"\b[a-zA-Z0-9.-]+\.(?:xyz|top|club|info|live|site|online|work|icu|vip|bid|monster|gq|cf|tk)\b",
]

PHISHING_LINK_PATTERNS = [
    r"\bclick (?:here|below|the link)\b",
    r"\bvisit (?:this |the )?link\b",
    r"\bverify using (?:this )?link\b",
    r"\bupdate using (?:this )?link\b",
]

IMPERSONATION_BRANDS = [
    "amazon", "paypal", "rbi", "reserve bank", "income tax", "sbi", "hdfc", "icici", 
    "axis", "paytm", "phonepe", "gpay", "google pay", "netflix", "fedex"
]


# ============================================================
# HELPER TO EXTRACT MATCHED TEXT SNIPPETS
# ============================================================

def find_matches(pattern_list, text_lower):
    matches = []
    for pat in pattern_list:
        m = re.search(pat, text_lower)
        if m:
            snippet = m.group(0).strip()
            if snippet not in matches:
                matches.append(snippet)
    return matches


# ============================================================
# PATTERN DETECTION ENGINE
# ============================================================

def detect_patterns(text):
    text_lower = text.lower()
    patterns = {}

    is_safety_warning = any(re.search(pat, text_lower) for pat in SAFETY_WARNING_PATTERNS)
    patterns["is_safety_warning"] = is_safety_warning

    is_maintenance_notice = any(re.search(pat, text_lower) for pat in MAINTENANCE_PATTERNS)
    patterns["is_maintenance_notice"] = is_maintenance_notice

    if is_safety_warning:
        patterns["account_threat"] = []
        patterns["sensitive_requests"] = {}
        patterns["sensitive_action"] = []
        patterns["action_request"] = []
        patterns["refund_bait"] = []
        patterns["advance_fee"] = []
        patterns["loan_scam"] = []
        patterns["investment_scam"] = []
        patterns["crypto_scam"] = []
        patterns["prize_bait"] = []
        patterns["upi_scam"] = []
        patterns["urgency"] = []
        patterns["scarcity"] = []
        patterns["urls"] = []
        patterns["phishing_link"] = []
        patterns["impersonation"] = []
        patterns["suspicious_combinations"] = []
        patterns["fees"] = []
        patterns["hidden_fees"] = []
        patterns["emotional"] = []
        patterns["threats"] = []
        patterns["link_action"] = []
        return patterns

    threats = find_matches(ACCOUNT_THREAT_PATTERNS, text_lower)
    patterns["account_threat"] = threats
    patterns["threats"] = threats

    sensitive_requests = {}
    for data_type, regex_list in SENSITIVE_INFO_PATTERNS.items():
        matched = find_matches(regex_list, text_lower)
        if matched:
            sensitive_requests[data_type] = matched
    patterns["sensitive_requests"] = sensitive_requests
    patterns["sensitive_action"] = list(sensitive_requests.keys())

    actions = find_matches(ACTION_REQUEST_PATTERNS, text_lower)
    patterns["action_request"] = actions

    refunds = find_matches(REFUND_PATTERNS, text_lower)
    patterns["refund_bait"] = refunds

    adv_fees = find_matches(ADVANCE_FEE_PATTERNS, text_lower)
    patterns["advance_fee"] = adv_fees
    patterns["fees"] = adv_fees
    patterns["hidden_fees"] = [f for f in adv_fees if "fee" in f]

    loan_scams = find_matches(LOAN_SCAM_PATTERNS, text_lower)
    patterns["loan_scam"] = loan_scams

    inv_scams = find_matches(INVESTMENT_SCAM_PATTERNS, text_lower)
    patterns["investment_scam"] = inv_scams

    crypto_scams = find_matches(CRYPTO_SCAM_PATTERNS, text_lower)
    patterns["crypto_scam"] = crypto_scams

    prizes = find_matches(PRIZE_PATTERNS, text_lower)
    patterns["prize_bait"] = prizes

    upi_scams = find_matches(UPI_SCAM_PATTERNS, text_lower)
    patterns["upi_scam"] = upi_scams

    urgency = find_matches(URGENCY_PATTERNS, text_lower)
    patterns["urgency"] = urgency

    scarcity = find_matches(SCARCITY_PATTERNS, text_lower)
    patterns["scarcity"] = scarcity
    patterns["emotional"] = scarcity

    urls = [url for pat in URL_PATTERNS for url in re.findall(pat, text)]
    patterns["urls"] = urls
    phish_links = find_matches(PHISHING_LINK_PATTERNS, text_lower)
    patterns["phishing_link"] = phish_links
    patterns["link_action"] = phish_links or urls

    matched_brands = [brand for brand in IMPERSONATION_BRANDS if brand in text_lower]
    patterns["impersonation"] = matched_brands

    combinations = detect_suspicious_combinations(patterns, text_lower)
    patterns["suspicious_combinations"] = combinations

    return patterns


# ============================================================
# SUSPICIOUS COMBINATION ENGINE
# ============================================================

def detect_suspicious_combinations(patterns, text_lower):
    combinations = []

    has_threat = bool(patterns.get("account_threat"))
    has_urgency = bool(patterns.get("urgency"))
    has_sensitive = bool(patterns.get("sensitive_requests"))
    has_action = bool(patterns.get("action_request"))
    has_link = bool(patterns.get("link_action"))
    has_refund = bool(patterns.get("refund_bait"))
    has_prize = bool(patterns.get("prize_bait"))
    has_advance_fee = bool(patterns.get("advance_fee"))
    has_loan = bool(patterns.get("loan_scam"))
    has_investment = bool(patterns.get("investment_scam"))
    has_crypto = bool(patterns.get("crypto_scam"))
    has_upi = bool(patterns.get("upi_scam"))
    has_brand = bool(patterns.get("impersonation"))

    # 1. Threat + Urgency + Action Request
    if has_threat and (has_urgency or has_action or has_sensitive or has_link):
        combinations.append("High-risk combination: Account suspension threat + urgency + action request")

    # 2. Refund + Sensitive Bank Details Request (e.g. Amazon refund scam)
    if has_refund and (has_sensitive or "bank account" in patterns.get("sensitive_requests", {}) or "routing number" in patterns.get("sensitive_requests", {})):
        combinations.append("High-risk combination: Refund claim + request for sensitive banking information")

    # 3. Refund + OTP Request
    if has_refund and ("otp" in patterns.get("sensitive_requests", {}) or "pin" in patterns.get("sensitive_requests", {})):
        combinations.append("Critical risk combination: Refund claim + OTP / PIN request")

    # 4. Prize + Advance Fee
    if has_prize and (has_advance_fee or has_urgency):
        combinations.append("High-risk combination: Prize / Lottery claim + upfront fee or urgency")

    # 5. Loan + Upfront Fee
    if has_loan and has_advance_fee:
        combinations.append("High-risk combination: Guaranteed loan offer + upfront processing fee")

    # 6. Investment + Guaranteed Return + Urgency
    if has_investment and (has_urgency or "scarcity" in patterns):
        combinations.append("High-risk combination: Guaranteed investment returns + high urgency / scarcity")

    # 7. QR Code / Receive Money + UPI PIN
    if has_upi or ("upi pin" in patterns.get("sensitive_requests", {}) and "receive" in text_lower):
        combinations.append("Critical risk combination: Request to enter UPI PIN or scan QR code to receive money")

    # 8. Crypto + Seed Phrase
    if has_crypto or "seed phrase" in patterns.get("sensitive_requests", {}):
        combinations.append("Critical risk combination: Crypto wallet verification / seed phrase request")

    # 9. Brand Impersonation + Refund / Threat + Data Request
    if has_brand and (has_refund or has_threat) and (has_sensitive or has_action):
        combinations.append("High-risk combination: Brand impersonation + refund/threat bait + sensitive data request")

    # 10. Link + Threat / Action
    if has_link and (has_threat or has_sensitive):
        combinations.append("High-risk combination: Suspicious link + account threat or credential request")

    return combinations


# ============================================================
# CATEGORY DETECTION
# ============================================================

def detect_offer_category(text):
    text_lower = text.lower()

    if any(re.search(pat, text_lower) for pat in SAFETY_WARNING_PATTERNS):
        return "Banking Security / KYC"

    if any(re.search(pat, text_lower) for pat in MAINTENANCE_PATTERNS):
        return "Account / Banking Alert"

    if any(k in text_lower for k in ["crypto", "bitcoin", "wallet", "airdrop", "seed phrase"]):
        return "Cryptocurrency"

    if any(k in text_lower for k in ["upi", "qr code", "upi pin", "collect request", "gpay", "phonepe", "paytm"]):
        return "UPI / Digital Payment"

    if any(k in text_lower for k in ["refund", "cashback", "reimbursement", "amazon refund", "tax refund"]):
        return "Refund / Payment"

    if any(k in text_lower for k in ["prize", "lottery", "won", "reward", "lucky customer"]):
        return "Prize / Reward"

    if any(k in text_lower for k in ["kyc", "pan", "aadhaar", "account suspended", "blocked", "deactivated"]):
        return "Banking Security / KYC"

    if any(k in text_lower for k in ["investment", "return", "profit", "monthly return", "ponzi", "stocks"]):
        return "Investment"

    if any(k in text_lower for k in ["loan", "emi", "borrow", "credit line", "mortgage"]):
        return "Loan"

    if any(k in text_lower for k in ["credit card", "debit card", "card limit", "card approval"]):
        return "Credit Card"

    if any(k in text_lower for k in ["deposit", "fixed deposit", "fd", "rd", "interest rate"]):
        return "Deposit"

    if any(k in text_lower for k in ["insurance", "policy", "premium"]):
        return "Insurance"

    return "Other"


# ============================================================
# MANIPULATION & TRANSPARENCY SCORING
# ============================================================

def calculate_manipulation_score(patterns):
    if patterns.get("is_safety_warning") or patterns.get("is_maintenance_notice"):
        return 0

    score = 0

    if patterns.get("account_threat"):
        score += 30
    if patterns.get("sensitive_requests"):
        score += 35
    if patterns.get("refund_bait"):
        score += 20
    if patterns.get("advance_fee"):
        score += 25
    if patterns.get("prize_bait"):
        score += 25
    if patterns.get("upi_scam"):
        score += 40
    if patterns.get("crypto_scam"):
        score += 40
    if patterns.get("investment_scam"):
        score += 30
    if patterns.get("loan_scam"):
        score += 25
    if patterns.get("urgency"):
        score += 15
    if patterns.get("scarcity"):
        score += 10
    if patterns.get("phishing_link") or patterns.get("urls"):
        score += 15

    combo_count = len(patterns.get("suspicious_combinations", []))
    score += combo_count * 20

    return min(score, 100)


def calculate_transparency_score(text, patterns, offer_category):
    if patterns.get("is_safety_warning") or patterns.get("is_maintenance_notice"):
        return 100

    score = 70

    if patterns.get("sensitive_requests"):
        score -= 40

    if patterns.get("suspicious_combinations"):
        score -= 30

    text_lower = text.lower()
    if offer_category == "Loan":
        if not any(k in text_lower for k in ["interest", "apr"]):
            score -= 15
        if not any(k in text_lower for k in ["terms", "conditions", "t&c"]):
            score -= 15
    elif offer_category == "Investment":
        if "guaranteed" in text_lower and "risk" not in text_lower:
            score -= 30

    return max(min(score, 100), 0)


# ============================================================
# HYBRID RISK DECISION ENGINE
# ============================================================

def get_risk_label(score):
    if score >= 80:
        return "Critical"
    elif score >= 60:
        return "High"
    elif score >= 40:
        return "Medium"
    elif score >= 20:
        return "Moderate"
    else:
        return "Low"


def hybrid_decision(ml_risk, manipulation_score, transparency_score, ethical_pred, patterns):
    if patterns.get("is_safety_warning") or patterns.get("is_maintenance_notice"):
        return 0, "Low", "Ethical"

    rule_score = (manipulation_score * 0.75) + ((100 - transparency_score) * 0.25)
    
    combinations = patterns.get("suspicious_combinations", [])
    has_critical_combo = any("Critical risk" in c for c in combinations)
    has_high_combo = any("High-risk" in c for c in combinations)

    if has_critical_combo:
        rule_score = max(rule_score, 85.0)
    elif has_high_combo:
        rule_score = max(rule_score, 70.0)

    final_score = int(round(rule_score))
    final_score = min(max(final_score, 0), 100)
    final_label = get_risk_label(final_score)

    if final_score >= 80:
        final_ethical = "Highly Manipulative"
    elif final_score >= 60:
        final_ethical = "Unethical / High Risk"
    elif final_score >= 40:
        final_ethical = "Potentially Manipulative"
    else:
        final_ethical = ethical_pred if ethical_pred in ["Ethical", "Unethical", "Potentially Manipulative"] else "Ethical"

    return final_score, final_label, final_ethical


# ============================================================
# CONCISE WARNING BULLETS (detected_patterns)
# ============================================================

def build_detected_patterns(patterns):
    warnings = []

    if patterns.get("is_safety_warning"):
        warnings.append("[SAFE] Official banking security advice detected (No risk).")
        return warnings

    if patterns.get("is_maintenance_notice"):
        warnings.append("[SAFE] Official service maintenance notice detected.")
        return warnings

    if patterns.get("account_threat"):
        warnings.append("[WARNING] Threat of account suspension/deactivation detected.")

    if patterns.get("sensitive_requests"):
        for req_type in patterns["sensitive_requests"].keys():
            warnings.append(f"[WARNING] Sensitive banking information requested: {req_type}")

    if patterns.get("refund_bait"):
        for ref in patterns["refund_bait"][:2]:
            warnings.append(f"[WARNING] Refund/payment bait detected: '{ref}'")

    if patterns.get("advance_fee"):
        warnings.append("[WARNING] Advance fee or upfront processing payment requested.")

    if patterns.get("prize_bait"):
        warnings.append("[WARNING] Prize / lottery winner bait detected.")

    if patterns.get("upi_scam"):
        warnings.append("[WARNING] Deceptive UPI PIN / QR code payment instruction detected.")

    if patterns.get("crypto_scam"):
        warnings.append("[WARNING] Cryptocurrency giveaway / wallet credential request detected.")

    if patterns.get("investment_scam"):
        warnings.append("[WARNING] Unrealistic guaranteed return / Ponzi investment claim detected.")

    if patterns.get("urgency"):
        warnings.append("[WARNING] High urgency or pressure language detected.")
    if patterns.get("scarcity"):
        warnings.append("[WARNING] Scarcity/limited time tactic detected.")

    if patterns.get("phishing_link") or patterns.get("urls"):
        warnings.append("[WARNING] Action request containing external link or URL detected.")

    for combo in patterns.get("suspicious_combinations", []):
        warnings.append(f"[ALERT] {combo}")

    if not warnings:
        warnings.append("[SAFE] No known manipulative or suspicious patterns detected.")

    return warnings


# ============================================================
# EXPLANATORY PATTERN INTERPRETATIONS (pattern_analysis)
# ============================================================

def build_pattern_analysis(patterns):
    """
    Generates rich, explanatory interpretations of detected fraud/manipulation patterns.
    Returns an EMPTY LIST [] if no meaningful pattern explanations exist.
    No fallback messages are appended if explanations is empty.
    """
    explanations = []

    # 1. Safety / Maintenance Notices
    if patterns.get("is_safety_warning"):
        return explanations

    if patterns.get("is_maintenance_notice"):
        return explanations

    # 2. Sensitive Data Requests
    if patterns.get("sensitive_requests"):
        data_items = list(patterns["sensitive_requests"].keys())
        items_str = " and ".join(data_items) if len(data_items) <= 2 else ", ".join(data_items[:3])
        explanations.append(
            f"Sensitive Data Request: The message asks for {items_str} information. "
            f"Financial identification details should never be shared through unsolicited messages or email/SMS replies."
        )

    # 3. Financial / Refund Baits
    if patterns.get("refund_bait"):
        explanations.append(
            "Financial Bait: A refund, payment reimbursement, or cashback claim is being used to encourage "
            "the recipient to respond, click links, or provide financial information."
        )

    if patterns.get("prize_bait"):
        explanations.append(
            "Financial Bait: A prize, lottery reward, or exclusive payout claim is used as bait "
            "to entice the recipient into taking immediate action or paying fees."
        )

    # 4. Fraud Pattern Combinations
    for combo in patterns.get("suspicious_combinations", []):
        if "Refund claim" in combo and "sensitive banking" in combo:
            explanations.append(
                "Fraud Pattern Combination: A refund claim combined with a request for sensitive banking information "
                "is a common social-engineering pattern."
            )
        elif "Account suspension threat" in combo:
            explanations.append(
                "Fraud Pattern Combination: An account threat combined with urgency and an action request "
                "is a classic phishing technique designed to provoke panic."
            )
        elif "Prize" in combo and "fee" in combo:
            explanations.append(
                "Fraud Pattern Combination: A prize notification combined with an upfront fee or urgency requirement "
                "is a classic advance-fee lottery scam."
            )
        elif "UPI PIN" in combo:
            explanations.append(
                "Fraud Pattern Combination: A refund or payment offer combined with a request to enter a UPI PIN or scan a QR code "
                "is a deceptive payment trap."
            )
        else:
            explanations.append(
                f"Fraud Pattern Combination: Multiple suspicious indicators ({combo}) appear together, "
                f"significantly increasing the likelihood of social engineering or financial fraud."
            )

    # 5. Brand Impersonation Risk
    if patterns.get("impersonation"):
        brands_str = ", ".join(b.capitalize() for b in patterns["impersonation"])
        explanations.append(
            f"Brand Impersonation Risk: The message references a well-known organization ({brands_str}). "
            f"The claimed organization should be verified independently before responding or sharing information."
        )

    # 6. External Link Analysis
    if patterns.get("urls") or patterns.get("phishing_link") or patterns.get("link_action"):
        explanations.append(
            "External Link Analysis: The message contains an external link or URL. "
            "The destination should be independently verified before entering credentials or financial information."
        )

    # 7. Account Threat
    if patterns.get("account_threat") and not any("Account Threat" in exp for exp in explanations):
        explanations.append(
            "Account Threat: The message creates fear by claiming that an account may be blocked, "
            "suspended, closed, or deactivated if immediate action is not taken."
        )

    # 8. Urgency Pressure
    if patterns.get("urgency"):
        explanations.append(
            "Urgency Pressure: The message attempts to create time pressure using phrases such as "
            "immediately, today, now, or urgent."
        )

    # 9. Advance Fee Requirement
    if patterns.get("advance_fee") and not any("advance-fee" in exp for exp in explanations):
        explanations.append(
            "Advance Fee Requirement: The message requests an upfront payment or processing fee "
            "before releasing promised funds, loans, or rewards."
        )

    # 10. UPI Scam Trap
    if patterns.get("upi_scam") and not any("UPI" in exp for exp in explanations):
        explanations.append(
            "UPI Fraud Trap: Instructing users to enter a UPI PIN or scan a QR code to receive money is deceptive; "
            "entering a UPI PIN always authorizes a payment/debit from your account."
        )

    # 11. Crypto Scams
    if patterns.get("crypto_scam"):
        explanations.append(
            "Crypto Credential Trap: Requesting wallet private keys, seed phrases, or wallet connections "
            "is a direct attempt to compromise digital asset wallets."
        )

    # 12. Investment Scams
    if patterns.get("investment_scam"):
        explanations.append(
            "Unrealistic Investment Promise: Claims of guaranteed high returns with zero risk "
            "are characteristic of fraudulent Ponzi or investment schemes."
        )

    # 13. Scarcity & Dark Patterns
    if patterns.get("scarcity"):
        explanations.append(
            "Scarcity Pressure: Limiting availability or claiming 'few slots left' is a psychological tactic "
            "designed to rush unconsidered decisions."
        )

    # If explanations is empty, return [] so UI can hide Pattern Analysis section entirely
    return explanations


# ============================================================
# RECOMMENDATION GENERATOR
# ============================================================

def generate_recommendation(risk_score, patterns):
    if patterns.get("is_safety_warning"):
        return "This is a legitimate safety notice. Always follow official security guidelines."

    if patterns.get("is_maintenance_notice"):
        return "This is an official service update. Plan your banking transactions accordingly."

    if patterns.get("sensitive_requests") or patterns.get("upi_scam") or patterns.get("crypto_scam"):
        return "CRITICAL WARNING: Never reply with your bank account number, routing number, OTP, PIN, or UPI PIN. EthosBank and legitimate organizations will never ask for secret credentials via email or SMS."

    if risk_score >= 80:
        return "DANGER: High probability of financial fraud or scam. Do not click any links, pay upfront fees, or share personal or financial details."

    if risk_score >= 60:
        return "WARNING: Suspicious offer detected. Verify the sender independently through official banking channels before taking any action."

    if risk_score >= 40:
        return "CAUTION: Review terms and conditions carefully before accepting or making payments."

    return "LOW RISK: This communication shows no strong manipulative or fraudulent indicators. Always verify official channels."


# ============================================================
# BUILD STRUCTURED FEATURES FOR ML MODELS
# ============================================================

def build_offer_features(text, patterns, transparency_score, manipulation_score, offer_category):
    primary_trigger = "None"
    if patterns.get("account_threat"):
        primary_trigger = "Account Threat"
    elif patterns.get("sensitive_requests"):
        primary_trigger = "Sensitive Action"
    elif patterns.get("refund_bait"):
        primary_trigger = "Refund Bait"
    elif patterns.get("urgency"):
        primary_trigger = "Urgency"

    secondary_trigger = "None"
    if patterns.get("urgency") and primary_trigger != "Urgency":
        secondary_trigger = "Urgency"
    elif patterns.get("link_action"):
        secondary_trigger = "Link Action"

    feature_dict = {
        "text": text,
        "interest_rate": float(0.0),
        "processing_fee": float(0.0),
        "loan_tenure": float(0.0),
        "contains_terms_and_conditions": int(1 if "terms" in text.lower() else 0),
        "mentions_processing_fee": int(1 if patterns.get("advance_fee") else 0),
        "mentions_interest_rate": int(1 if "interest" in text.lower() else 0),
        "mentions_penalty": int(1 if "penalty" in text.lower() else 0),
        "mentions_foreclosure": int(1 if "foreclosure" in text.lower() else 0),
        "urgency_phrase": int(1 if patterns.get("urgency") else 0),
        "scarcity_phrase": int(1 if patterns.get("scarcity") else 0),
        "emotional_phrase": int(1 if patterns.get("emotional") else 0),
        "hidden_fee_indicator": int(1 if patterns.get("hidden_fees") else 0),
        "transparency_score": float(transparency_score),
        "manipulation_score": float(manipulation_score),
        "complexity_score": float(0.0),
        "institution_type": "Unknown",
        "product": offer_category,
        "primary_trigger": primary_trigger,
        "secondary_trigger": secondary_trigger,
        "offer_category": offer_category,
    }

    df = pd.DataFrame([feature_dict])

    target_model = ethical_model or risk_model
    if target_model and hasattr(target_model, "feature_names_in_"):
        expected_cols = getattr(target_model, "feature_names_in_")
        df = df.reindex(columns=expected_cols, fill_value=0.0)

    for num_col in ["interest_rate", "processing_fee", "loan_tenure", "complexity_score", "transparency_score", "manipulation_score"]:
        if num_col in df.columns:
            df[num_col] = df[num_col].astype(float)

    return df


# ============================================================
# MAIN OFFER ANALYSIS ENGINE
# ============================================================

def analyze_offer(offer_text):
    print("\n" + "=" * 75)
    print("STARTING ETHOSBANK AI HYBRID OFFER SCAN V3")
    print("=" * 75)

    if not offer_text or not offer_text.strip():
        raise ValueError("Offer text cannot be empty.")

    # 1. Clean Text
    cleaned_text = clean_text(offer_text)

    # 2. Detect Category & Patterns
    offer_category = detect_offer_category(cleaned_text)
    patterns = detect_patterns(cleaned_text)

    # 3. Calculate Rule Scores
    manipulation_score = calculate_manipulation_score(patterns)
    transparency_score = calculate_transparency_score(cleaned_text, patterns, offer_category)

    # 4. Run ML Models
    sentiment = safe_predict_sentiment(cleaned_text)
    banking_intent = safe_predict_banking_intent(cleaned_text)

    offer_features_df = build_offer_features(
        cleaned_text, patterns, transparency_score, manipulation_score, offer_category
    )

    ethical_pred = safe_predict_ethical(offer_features_df, manipulation_score)
    ml_risk_pred = safe_predict_ml_risk(offer_features_df)

    # 5. Hybrid Final Decision
    final_risk_score, final_risk_label, final_ethical = hybrid_decision(
        ml_risk_pred, manipulation_score, transparency_score, ethical_pred, patterns
    )

    # 6. Process Summary, Warnings & Explanatory Pattern Analysis
    processed_text = distilbart_process(offer_text, cleaned_text, patterns, offer_category)
    detected_patterns = build_detected_patterns(patterns)
    pattern_analysis = build_pattern_analysis(patterns)
    recommendation = generate_recommendation(final_risk_score, patterns)

    result = {
        "original_text": offer_text,
        "cleaned_text": processed_text,
        "processed_text": processed_text,
        "sentiment": sentiment,
        "banking_intent": banking_intent,
        "ethical_classification": str(final_ethical),
        "ml_risk_prediction": str(ml_risk_pred),
        "risk_level": final_risk_label,
        "risk_score": final_risk_score,
        "manipulation_score": manipulation_score,
        "transparency_score": transparency_score,
        "offer_category": offer_category,
        "detected_patterns": detected_patterns,
        "pattern_analysis": pattern_analysis,
        "pattern_details": patterns,
        "recommendation": recommendation,
    }

    print("\n" + "=" * 75)
    print("ETHOSBANK AI - FINAL HYBRID RESULT")
    print("=" * 75)
    print(f"Offer Category:     {result['offer_category']}")
    print(f"Risk Level:         {result['risk_level']} ({result['risk_score']}/100)")
    print(f"Ethical Status:     {result['ethical_classification']}")
    print(f"ML Risk Prediction: {result['ml_risk_prediction']}")
    print(f"Manipulation Score: {result['manipulation_score']}/100")
    print(f"Transparency Score: {result['transparency_score']}/100")
    print("\nDetected Patterns:")
    for warn in detected_patterns:
        print(f"    {warn}")
    print("\nExplanatory Pattern Analysis:")
    for exp in pattern_analysis:
        print(f"    - {exp}")
    print(f"\nRecommendation:\n    {recommendation}")
    print("=" * 75)

    return result