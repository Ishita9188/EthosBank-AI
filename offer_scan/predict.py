import os
import joblib
import pandas as pd

MODEL_DIR = os.path.join(os.path.dirname(__file__), "ml_models")

class OfferScanPredictorV3:
    def __init__(self):
        print("Loading Version 3 models...")
        self.sentiment_model = joblib.load(os.path.join(MODEL_DIR, "financial_sentiment_model_v3.pkl"))
        self.intent_model = joblib.load(os.path.join(MODEL_DIR, "banking_intent_model_v3.pkl"))
        self.ethical_model = joblib.load(os.path.join(MODEL_DIR, "ethical_offer_model_v3.pkl"))
        self.risk_model = joblib.load(os.path.join(MODEL_DIR, "offer_risk_model_v3.pkl"))
        print("All V3 models loaded successfully.")

    def scan_offer(self, offer_data: dict) -> dict:
        """
        Expects a dictionary with text and numeric/categorical features for custom offer analysis.
        """
        text_df = pd.DataFrame([{"text": offer_data.get("text", "")}])
        
        # Predict Sentiment and Intent from raw text
        sentiment = self.sentiment_model.predict(text_df)[0]
        intent = self.intent_model.predict(text_df)[0]
        
        # Format full feature row for Ethical & Risk models
        full_df = pd.DataFrame([offer_data])
        
        ethical_label = self.ethical_model.predict(full_df)[0]
        risk_level = self.risk_model.predict(full_df)[0]
        
        return {
            "financial_sentiment": sentiment,
            "banking_intent": intent,
            "ethical_evaluation": ethical_label,
            "offer_risk_level": risk_level
        }

if __name__ == "__main__":
    predictor = OfferScanPredictorV3()
    
    # Test sample
    sample_offer = {
        "text": "Pre-approved personal loan at 10.5% interest rate. Apply now with low processing fee!",
        "interest_rate": 10.5,
        "processing_fee": 500,
        "loan_tenure": 36,
        "contains_terms_and_conditions": 1,
        "mentions_processing_fee": 1,
        "mentions_interest_rate": 1,
        "mentions_penalty": 0,
        "mentions_foreclosure": 0,
        "urgency_phrase": 1,
        "scarcity_phrase": 0,
        "emotional_phrase": 0,
        "hidden_fee_indicator": 0,
        "transparency_score": 8,
        "manipulation_score": 2,
        "complexity_score": 3,
        "institution_type": "Bank",
        "product": "Personal Loan",
        "primary_trigger": "Urgency",
        "secondary_trigger": "Pre-Approval",
        "offer_category": "Credit"
    }
    
    result = predictor.scan_offer(sample_offer)
    print("\nScan Result:", result)