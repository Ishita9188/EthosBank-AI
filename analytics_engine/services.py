from pyexpat import features
from sklearn import cluster
from sklearn.linear_model import LinearRegression
import os
import joblib
from datetime import date, timedelta
import numpy as np
import pandas as pd
from django.utils import timezone
from django.db.models import Sum
import shap
import plotly.graph_objects as go
from accounts.models import UserAccount, FinancialProfile
from banking.models import Wallet, Transaction, FixedDeposit, RecurringDeposit, PassionFund
from .models import AIInsight, AIExplanation

_models_cache = {}

def get_saved_model(filename):
    """Loads a model from saved_models directory and caches it in memory."""
    if filename not in _models_cache:
        path = os.path.join('analytics_engine', 'saved_models', filename)
        if os.path.exists(path):
            try:
                _models_cache[filename] = joblib.load(path)
            except Exception as e:
                print(f"Error loading model {filename}: {e}")
                return None
        else:
            print(f"Model file {path} does not exist.")
            return None
    return _models_cache[filename]
def generate_shap_explanation(model, features, feature_names):
    """
    Generate SHAP explanations for a single prediction.
    Returns top 6 feature contributions for class 1.
    """

    print("\n========== SHAP DEBUG ==========")

    try:

        # ---------------------------------------------------------
        # 1. Check model
        # ---------------------------------------------------------
        if model is None:
            print("SHAP ERROR: Model is None")
            return []

        print("Model Type:", type(model).__name__)
        print("Number of Features:", len(features))
        print("Number of Feature Names:", len(feature_names))

        print("Features:", features)
        print("Feature Names:", feature_names)

        # ---------------------------------------------------------
        # 2. Create DataFrame
        # ---------------------------------------------------------
        X = pd.DataFrame(
            [features],
            columns=feature_names
        )

        print("SHAP DataFrame:")
        print(X)

        # ---------------------------------------------------------
        # 3. Create TreeExplainer
        # ---------------------------------------------------------
        print("Creating TreeExplainer...")

        explainer = shap.TreeExplainer(model)

        print("TreeExplainer created successfully.")

        # ---------------------------------------------------------
        # 4. Calculate SHAP values
        # ---------------------------------------------------------
        print("Calculating SHAP values...")

        shap_values = explainer.shap_values(X)

        print(
            "SHAP raw type:",
            type(shap_values)
        )

        if isinstance(shap_values, list):

            print(
                "SHAP list length:",
                len(shap_values)
            )

            for i, arr in enumerate(shap_values):
                print(
                    f"SHAP class {i} shape:",
                    np.asarray(arr).shape
                )

        else:

            print(
                "SHAP array shape:",
                np.asarray(shap_values).shape
            )

        # ---------------------------------------------------------
        # 5. Handle SHAP output
        # ---------------------------------------------------------

        if isinstance(shap_values, list):

            # Binary classification
            if len(shap_values) > 1:

                values = np.asarray(
                    shap_values[1]
                )

                if values.ndim == 2:
                    values = values[0]

            else:

                values = np.asarray(
                    shap_values[0]
                )

                if values.ndim == 2:
                    values = values[0]

        else:

            values = np.asarray(shap_values)

            # New SHAP format:
            # samples × features × classes
            if values.ndim == 3:

                print(
                    "Detected 3D SHAP output."
                )

                if values.shape[2] > 1:

                    values = values[0, :, 1]

                else:

                    values = values[0, :, 0]

            # samples × features
            elif values.ndim == 2:

                print(
                    "Detected 2D SHAP output."
                )

                values = values[0]

            # features
            elif values.ndim == 1:

                print(
                    "Detected 1D SHAP output."
                )

            else:

                values = values.flatten()

        # ---------------------------------------------------------
        # 6. Convert to 1D
        # ---------------------------------------------------------

        values = np.asarray(
            values,
            dtype=float
        ).flatten()

        print(
            "Final SHAP values:",
            values
        )

        print(
            "SHAP value count:",
            len(values)
        )

        # ---------------------------------------------------------
        # 7. Validate
        # ---------------------------------------------------------

        if len(values) != len(feature_names):

            print(
                "SHAP ERROR: Feature count mismatch!"
            )

            print(
                "Expected:",
                len(feature_names)
            )

            print(
                "Received:",
                len(values)
            )

            return []

        # ---------------------------------------------------------
        # 8. Create explanations
        # ---------------------------------------------------------

        explanations = []

        for name, value, shap_value in zip(
            feature_names,
            features,
            values
        ):

            shap_value = float(shap_value)

            explanations.append({

                'feature': name,

                'value': round(
                    float(value),
                    4
                ),

                'shap_value': round(
                    shap_value,
                    4
                ),

                'absolute_shap': round(
                    abs(shap_value),
                    4
                ),

                'impact': (
                    'Increases Risk'
                    if shap_value > 0
                    else
                    'Decreases Risk'
                    if shap_value < 0
                    else
                    'No Significant Impact'
                )

            })

        # ---------------------------------------------------------
        # 9. Sort
        # ---------------------------------------------------------

        explanations.sort(
            key=lambda x: x['absolute_shap'],
            reverse=True
        )

        # ---------------------------------------------------------
        # 10. Return top 6
        # ---------------------------------------------------------

        result = explanations[:6]

        print("Top SHAP explanations:")

        for item in result:
            print(
                item['feature'],
                "=>",
                item['shap_value'],
                item['impact']
            )

        print("========== SHAP SUCCESS ==========\n")

        return result

    except Exception as e:

        print("\n========== SHAP ERROR ==========")

        print(
            "Error Type:",
            type(e).__name__
        )

        print(
            "Error Message:",
            str(e)
        )

        import traceback

        traceback.print_exc()

        print("=================================\n")

        return []

class FinancialWellnessService:

    @staticmethod
    def get_or_calculate_insights(user, force_recalc=False):
        """
        Retrieves cached insights or triggers recalculation.
        Returns a dict of insights and explanations.
        """
        insight_types = ['Financial Wellness', 'Debt Risk', 'Savings Forecast', 'Impulse Spending', 'Poverty Risk']
        
        # Check if we have cached insights for all types
        existing_insights = AIInsight.objects.filter(user=user)
        existing_explanations = AIExplanation.objects.filter(user=user)
        
        # Cache hit if we have all insights, explanations and force_recalc is False
        if not force_recalc and existing_insights.count() >= 5 and existing_explanations.count() >= 5:
            insights_dict = {}
            for ins in existing_insights:
                insights_dict[ins.insight_type] = {
                    'score': ins.score,
                    'recommendation': ins.recommendation,
                }
            for exp in existing_explanations:
                # Add explanation details
                if exp.model_name in insights_dict or exp.model_name == 'Spending Pattern':
                    key = 'Impulse Spending' if exp.model_name == 'Spending Pattern' else exp.model_name
                    if key in insights_dict:
                        insights_dict[key]['prediction'] = exp.prediction
                        insights_dict[key]['confidence'] = exp.confidence
                        insights_dict[key]['explanation'] = exp.explanation
                        saved_features = exp.feature_importance or {}

                        shap_explanation = saved_features.pop(
                            '__shap__',
                            []
                        )

                        insights_dict[key]['feature_importance'] = saved_features
                        insights_dict[key]['shap_explanation'] = shap_explanation
            
            # Additional double check to make sure everything was populated
            if all(k in insights_dict and 'explanation' in insights_dict[k] for k in insight_types):
                return insights_dict

        # Cache miss: recalculate and cache results
        return FinancialWellnessService.recalculate_insights(user)

    @staticmethod
    def recalculate_insights(user):
        """
        Loads user details, extracts features, runs rule engines & ML models,
        saves outputs, and returns structured result.
        """
        print(f"Recalculating AI insights for user {user.customer_id}...")
        
        # Clear existing cached values first to keep database clean
        AIInsight.objects.filter(user=user).delete()
        AIExplanation.objects.filter(user=user).delete()

        # Retrieve wallet
        try:
            wallet = Wallet.objects.get(user=user)
            wallet_balance = float(wallet.balance)
        except Wallet.DoesNotExist:
            wallet_balance = 0.0

        # Retrieve user profile
        try:
            profile = FinancialProfile.objects.get(user=user)
            profile_exists = True
        except FinancialProfile.DoesNotExist:
            profile_exists = False
            profile = None

        # Gather active banking products
        active_fds = FixedDeposit.objects.filter(user=user, status='Active')
        total_fd_balance = float(active_fds.aggregate(Sum('amount'))['amount__sum'] or 0.0)

        active_rds = RecurringDeposit.objects.filter(user=user, status='Active')
        total_rd_monthly = float(active_rds.aggregate(Sum('monthly_amount'))['monthly_amount__sum'] or 0.0)
        total_rd_paid = float(active_rds.aggregate(Sum('installments_paid'))['installments_paid__sum'] or 0.0) # we'll estimate RD balance
        # Estimate RD balance as installments_paid * monthly_amount
        total_rd_balance = 0.0
        for rd in active_rds:
            total_rd_balance += float(rd.monthly_amount) * rd.installments_paid

        active_passion_funds = PassionFund.objects.filter(user=user, status='Active')
        total_passion_balance = float(active_passion_funds.aggregate(Sum('current_amount'))['current_amount__sum'] or 0.0)
        total_passion_monthly = float(active_passion_funds.aggregate(Sum('monthly_installment'))['monthly_installment__sum'] or 0.0)
        total_passion_target = float(active_passion_funds.aggregate(Sum('target_amount'))['target_amount__sum'] or 0.0)

        # Retrieve user transactions (last 30 days and all-time)
        now = timezone.now()
        last_30_days = now - timedelta(days=30)
        recent_tx = Transaction.objects.filter(user=user, created_at__gte=last_30_days)
        all_tx = Transaction.objects.filter(user=user)

        total_deposits_30 = float(recent_tx.filter(transaction_type='Deposit').aggregate(Sum('amount'))['amount__sum'] or 0.0)
        total_withdrawals_30 = float(recent_tx.filter(transaction_type='Withdraw').aggregate(Sum('amount'))['amount__sum'] or 0.0)
        total_transfers_30 = float(recent_tx.filter(transaction_type='Transfer').aggregate(Sum('amount'))['amount__sum'] or 0.0)
        total_expenses_30 = float(total_withdrawals_30 + total_transfers_30)

        # Base case fallback if profile doesn't exist
        if not profile_exists:
            # Create a graceful warning explanation and default values
            default_wellness = {
                'score': 50.0,
                'recommendation': "Complete your Financial Profile in the 'Financial Profile' tab to generate custom AI reports.",
                'prediction': 'Average',
                'confidence': 50.0,
                'explanation': "A default baseline score is applied because your Financial Profile is empty. Completing your profile allows our AI engine to analyze your metrics.",
                'feature_importance': {},
                'shap_explanation': []
            }
            default_debt = {
                'score': 0.0,
                'recommendation': "Complete your Financial Profile to enable debt risk prediction.",
                'prediction': 'Low',
                'confidence': 90.0,
                'explanation': "No active liabilities or financial profile found. Defaulting to Low Risk.",
                'feature_importance': {},
                'shap_explanation': []
            }
            default_savings = {
                'score': wallet_balance,
                'recommendation': "Start saving or open a Recurring Deposit / Passion Fund to forecast future assets.",
                'prediction': f"₹{wallet_balance}",
                'confidence': 80.0,
                'explanation': "Savings forecast is flat based on current wallet balance because there is no profile history.",
                'feature_importance': {'3_months': wallet_balance, '6_months': wallet_balance, '12_months': wallet_balance}
            }
            default_spending = {
                'score': 0.0,
                'recommendation': "Begin transacting using your wallet to start tracking spending categories.",
                'prediction': 'Balanced',
                'confidence': 100.0,
                'explanation': "No transaction history detected. Begin transacting to map your spending patterns.",
                'feature_importance': {'top_category': 'N/A', 'monthly_trend': 'Stable', 'weekly_trend': 'Stable', 'increasing_expenses': 'None', 'decreasing_expenses': 'None'}
            }
            default_poverty = {
                'score': 0.0,
                'recommendation': "Complete your Financial Profile to analyze your safety net indicators.",
                'prediction': 'Low',
                'confidence': 100.0,
                'explanation': "Low poverty risk due to empty financial details. Fill profile to enable check.",
                'feature_importance': {}
            }
            
            # Save defaults to DB
            FinancialWellnessService.save_insight_db(user, 'Financial Wellness', default_wellness)
            FinancialWellnessService.save_insight_db(user, 'Debt Risk', default_debt)
            FinancialWellnessService.save_insight_db(user, 'Savings Forecast', default_savings)
            FinancialWellnessService.save_insight_db(user, 'Impulse Spending', default_spending, 'Spending Pattern')
            FinancialWellnessService.save_insight_db(user, 'Poverty Risk', default_poverty)
            
            return {
                'Financial Wellness': default_wellness,
                'Debt Risk': default_debt,
                'Savings Forecast': default_savings,
                'Impulse Spending': default_spending,
                'Poverty Risk': default_poverty
            }

        # Profile exists, extract profile variables
        monthly_income = float(profile.monthly_income) + float(profile.other_monthly_income or 0.0)
        if monthly_income <= 0:
            monthly_income = 1.0  # Avoid division by zero
            
        monthly_savings_target = float(profile.monthly_savings_target or 0.0)
        preferred_monthly_budget = float(profile.preferred_monthly_budget or 0.0)
        dependents = int(profile.dependents or 0)
        # Wallet Balance
        wallet_balance = 0
        wallet = Wallet.objects.filter(user=user).first()

        if wallet:
            wallet_balance = float(wallet.balance or 0)


        # Active Fixed Deposits
        fd_total = float(
            FixedDeposit.objects
            .filter(user=user, status="Active")
            .aggregate(total=Sum("amount"))["total"]
            or 0
        )


        # Active Recurring Deposits
        # Recurring Deposits
        rd_total = 0.0
        for rd in RecurringDeposit.objects.filter(user=user, status="Active"):
            rd_total += float(rd.monthly_amount) * rd.installments_paid


        # Active Passion Funds
        passion_total = float(
            PassionFund.objects
            .filter(user=user, status="Active")
            .aggregate(total=Sum("current_amount"))["total"]
            or 0
        )
        # Final Savings
        existing_savings = (
            wallet_balance
            + fd_total
            + rd_total
            + passion_total
        )
        employment_type = profile.employment_type or "Unknown"
        home_loan = float(profile.home_loan or 0)
        education_loan = float(profile.education_loan or 0)
        personal_loan = float(profile.personal_loan or 0)
        credit_card_debt = float(profile.credit_card_debt or 0)
        other_debt = float(profile.other_debt or 0)
        
        total_debt = home_loan + education_loan + personal_loan + credit_card_debt + other_debt
        # STEP 1: Compute Rule-Based Wellness score (Savings, Expense, Debt, Emergency, Budget, Goals)
        # A. Savings Ratio
        # RDs, Passion Funds and target savings are all assets/savings
        # =====================================================
# SAVINGS ANALYSIS (Hybrid Approach)
# =====================================================

# Monthly saving behaviour
        monthly_saved = (
        monthly_savings_target +
        total_rd_monthly +
        total_passion_monthly
    )

        if monthly_income > 0:
            monthly_saving_ratio = monthly_saved / (monthly_income * 0.20)
        else:
            monthly_saving_ratio = 0

        monthly_saving_ratio = min(monthly_saving_ratio, 1.0)

# -----------------------------------------------------
# Total accumulated savings
# -----------------------------------------------------

        rd_saved = float(
            Transaction.objects.filter(
                user=user,
                transaction_type="RD Installment"
            ).aggregate(total=Sum("amount"))["total"] or 0
        )

        total_actual_savings = existing_savings

        # Emergency fund target = 3 months of income
        if monthly_income > 0:
            emergency_ratio = total_actual_savings / (monthly_income * 3)
        else:
            emergency_ratio = 0

        emergency_ratio = min(emergency_ratio, 1.0)

        # -----------------------------------------------------
        # Final Hybrid Savings Score
        # 40% Monthly Saving Behaviour
        # 60% Emergency Savings Strength
        # -----------------------------------------------------

        savings_score = (
            (monthly_saving_ratio * 40) +
            (emergency_ratio * 60)
)

        # Optional values for dashboard
        savings_ratio = emergency_ratio
        # 20% savings ratio gets 100 points
        # B. Expense Ratio
        # Real expenses in last 30 days vs monthly income
        expenses_to_evaluate = total_expenses_30 if all_tx.exists() else preferred_monthly_budget
        expense_ratio = expenses_to_evaluate / monthly_income
        # Target: spend 50% or less. Decay score linearly for spending over 50%.
        if expense_ratio <= 0.50:
            expense_score = 100.0
        else:
            expense_score = max(0.0, 100.0 - (expense_ratio - 0.50) * 200.0)
        # C. Debt Ratio
        # Ratio of total outstanding debt to annual income
        debt_ratio = total_debt / (monthly_income * 12.0)
        # Target: debt is 40% or less of annual income.
        if debt_ratio <= 0.40:
            debt_score = 100.0 - (debt_ratio / 0.40) * 50.0 # Capped at 50 to 100
        else:
            debt_score = max(0.0, 50.0 - ((debt_ratio - 0.40) / 0.60) * 50.0)

        # D. Emergency Savings
        # Liquid funds (wallet balance + existing savings) in terms of months of income
        liquid_funds = existing_savings
        emergency_months = liquid_funds / monthly_income
        low_emergency_fund = emergency_months < 3
        # Target: 3 months of income.
        emergency_score = min((emergency_months / 3.0) * 100.0, 100.0)

        # E. Monthly Budget Usage
        # Real expenses vs preferred monthly budget
        if preferred_monthly_budget > 0:
            budget_usage = expenses_to_evaluate / preferred_monthly_budget
            if budget_usage <= 1.0:
                budget_score = 100.0
            else:
                budget_score = max(0.0, 100.0 - (budget_usage - 1.0) * 100.0) # lose points linearly for exceeding
        else:
            budget_score = 100.0

        # F. Financial Goal Progress
        # Passion funds, fixed deposits and recurring deposits progress
        goal_saved = total_passion_balance + total_fd_balance + total_rd_balance
        goal_target = total_passion_target + total_fd_balance + total_rd_balance
        if goal_target > 0:
            goal_progress = goal_saved / goal_target
            goal_score = min(goal_progress * 100.0, 100.0)
        else:
            goal_score = 80.0 # baseline if no specific goals set

        # Compute Rule Score (Weighted Average)
        rule_score = (
            savings_score * 0.20 +
            expense_score * 0.20 +
            debt_score * 0.20 +
            emergency_score * 0.20 +
            budget_score * 0.10 +
            goal_score * 0.10
        )

        # ---------------------------------------------------------------------
        # STEP 2: ML-Based Wellness Score (Random Forest)
        # ---------------------------------------------------------------------
        rf_wellness = get_saved_model('wellness_rf.pkl')
        
        # Prepare feature vector (must match order of columns in cs-training.csv)
        # 1. RevolvingUtilizationOfUnsecuredLines (proxy: credit_card_debt / (income * 2 + 1))
        revol_util = credit_card_debt / (monthly_income * 2.0 + 1000.0)
        revol_util = min(revol_util, 1.0)
        
        # 2. age
        age = (date.today() - user.dob).days // 365
        
        # 3. NumberOfTime30-59DaysPastDueNotWorse (Check RD next due dates)
        past_due_30 = 0
        past_due_60 = 0
        past_due_90 = 0
        
        # Let's count if any active recurring deposit is overdue
        today_val = date.today()
        for rd in active_rds:
            if rd.next_due_date < today_val:
                days_overdue = (today_val - rd.next_due_date).days
                if 30 <= days_overdue < 60:
                    past_due_30 += 1
                elif 60 <= days_overdue < 90:
                    past_due_60 += 1
                elif days_overdue >= 90:
                    past_due_90 += 1
                    
        # 4. DebtRatio (Monthly debt payments proxy: 1% of total debt / monthly income)
        monthly_debt_pay = total_debt * 0.01
        ml_debt_ratio = monthly_debt_pay / monthly_income
        
        # 5. MonthlyIncome
        ml_income = monthly_income
        
        # 6. NumberOfOpenCreditLinesAndLoans
        num_open_loans = 0
        for loan in [home_loan, education_loan, personal_loan, credit_card_debt, other_debt]:
            if loan > 0:
                num_open_loans += 1
                
        # 7. NumberOfTimes90DaysLate
        # 8. NumberRealEstateLoansOrLines
        num_re_loans = 1 if home_loan > 0 else 0
        
        # 9. NumberOfTime60-89DaysPastDueNotWorse
        # 10. NumberOfDependents
        num_dependents = dependents

        feature_names = [
    'Revolving Utilization',
    'Age',
    '30-59 Days Past Due',
    'Debt Ratio',
    'Monthly Income',
    'Open Credit Lines',
    '90+ Days Past Due',
    'Real Estate Loans',
    '60-89 Days Past Due',
    'Number of Dependents'
]

        features = [
    revol_util,
    age,
    past_due_30,
    ml_debt_ratio,
    ml_income,
    num_open_loans,
    past_due_90,
    num_re_loans,
    past_due_60,
    num_dependents
]

        if rf_wellness is not None:
            # Predict default probability
            try:
                feature_names = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents"
]

                features_df = pd.DataFrame([features], columns=feature_names)

                prob_default = rf_wellness.predict_proba(features_df)[0][1]
                ml_score = (1.0 - prob_default) * 100.0
                wellness_confidence = float(np.max(rf_wellness.predict_proba(features_df)[0])) * 100.0
            except Exception as e:
                print(f"Error predicting wellness: {e}")
                ml_score = rule_score
                wellness_confidence = 75.0
        else:
            ml_score = rule_score
            wellness_confidence = 70.0

        # Combine scores (50% rule-based, 50% ML)
        final_wellness_score = round(0.5 * rule_score + 0.5 * ml_score, 1)

        # Wellness rating
        if final_wellness_score >= 80:
            wellness_rating = 'Excellent'
        elif final_wellness_score >= 60:
            wellness_rating = 'Good'
        elif final_wellness_score >= 40:
            wellness_rating = 'Average'
        else:
            wellness_rating = 'Poor'

        # Generate Wellness Explanation
        wellness_reasons = []
        if emergency_ratio < 0.15:
            wellness_reasons.append("Savings ratio is below the recommended 15% threshold.")
        else:
            wellness_reasons.append("Maintains a healthy savings ratio.")
            
        if expense_ratio > 0.60:
            wellness_reasons.append("High monthly expenses relative to monthly income.")
        
        if emergency_months < 3.0:
            wellness_reasons.append("Emergency fund reserves are below the 3-month safety margin.")
        else:
            wellness_reasons.append("Strong emergency fund cushion established.")
            
        if total_debt > 0 and debt_ratio > 0.35:
            wellness_reasons.append("Debt burden takes up a significant portion of income.")
            
        if past_due_30 > 0 or past_due_90 > 0:
            wellness_reasons.append("System detected past due payments on deposits/loans.")

        wellness_explanation = " ".join(wellness_reasons)
        if not wellness_explanation:
            wellness_explanation = "Your finances are balanced and in good order."

        # Recommendations list for wellness
        wellness_rec_list = []
        if emergency_ratio < 0.15:
            wellness_rec_list.append("Increase your monthly savings target by ₹2,000.")
        if emergency_months < 3.0:
            wellness_rec_list.append("Build emergency fund to cover at least 3 months of expenses.")
        if expense_ratio > 0.60:
            wellness_rec_list.append("Reduce shopping or dining expenses to balance budget.")
        if total_debt > 0 and debt_ratio > 0.35:
            wellness_rec_list.append("Prioritize paying off high-interest credit card debt.")
        if not wellness_rec_list:
            wellness_rec_list.append("Maintain current excellent spending and savings habits.")
            
        wellness_rec = "; ".join(wellness_rec_list)
        # ============================================================
# SHAP EXPLAINABLE AI - FINANCIAL WELLNESS
# ============================================================

        wellness_shap = generate_shap_explanation(
            rf_wellness,
            features,
            feature_names
        )
        wellness_feat_imp = {

    'Monthly Income': round(monthly_income,2),

    'Wallet Balance': round(wallet_balance,2),
    'FD Balance': round(fd_total,2),
    'RD Balance': round(rd_total,2),
    'Passion Fund Balance': round(passion_total,2),

    'Total Savings': round(existing_savings,2),

    'Monthly Expenses': round(total_expenses_30,2),

    'Savings Ratio': round(monthly_saving_ratio,2),

    'Expense Ratio': round(expense_ratio,2),

    'Debt Ratio': round(debt_ratio,2),

    'Emergency Months': round(emergency_months,2),

    'Budget Usage': round(budget_usage if preferred_monthly_budget > 0 else 0,2),

    'Goal Progress': round(goal_progress if goal_target>0 else 0,2),

    'Savings Score': round(savings_score,1),

    'Expense Score': round(expense_score,1),

    'Debt Score': round(debt_score,1),

    'Emergency Score': round(emergency_score,1),

    'Budget Score': round(budget_score,1),

    'Goal Score': round(goal_score,1),

    'ML Probability': round(ml_score,1),
    # SHAP explanation
    '__shap__': wellness_shap
}

        wellness_data = {
            'score': float(final_wellness_score),
            'recommendation': wellness_rec,
            'prediction': wellness_rating,
            'confidence': float(round(wellness_confidence, 1)),
            'explanation': wellness_explanation,
            'feature_importance': wellness_feat_imp,
            'shap_explanation': wellness_shap
        }
        FinancialWellnessService.save_insight_db(user, 'Financial Wellness', wellness_data)

        # ---------------------------------------------------------------------
        # STEP 3: Debt Risk Prediction (HYBRID - Random Forest + XGBoost)
        # ---------------------------------------------------------------------
        rf_debt = get_saved_model('debt_rf.pkl')
        xgb_debt = get_saved_model('debt_xgb.pkl')
        # ============================================================
# SHAP EXPLAINABLE AI - DEBT RISK
# ============================================================

        debt_shap_rf = generate_shap_explanation(
            rf_debt,
            features,
            feature_names
        )

        debt_shap_xgb = generate_shap_explanation(
            xgb_debt,
            features,
            feature_names
        )
        
        p_rf = 0.0
        p_xgb = 0.0
        
        # Default probabilities using models
        print("\n========== DEBT MODEL DEBUG ==========")
        print("RF Model Loaded :", rf_debt is not None)
        print("XGB Model Loaded:", xgb_debt is not None)
        print("Features:", features)

        if rf_debt is not None and xgb_debt is not None:
            try:
                features_arr = np.nan_to_num(np.array([features], dtype=float))

                p_rf = float(rf_debt.predict_proba(features_arr)[0][1])
                p_xgb = float(xgb_debt.predict_proba(features_arr)[0][1])

                print("RF Probability :", p_rf)
                print("XGB Probability:", p_xgb)

            except Exception as e:
                print("Debt Prediction Error:", e)

                p_rf = ml_debt_ratio
                p_xgb = ml_debt_ratio

        else:
            print("Debt models not loaded.")

            p_rf = ml_debt_ratio
            p_xgb = ml_debt_ratio

        print("======================================")

        # Determine risk levels
        def get_risk_level(p):
            if p < 0.12: return 'Low'
            elif p < 0.35: return 'Medium'
            else: return 'High'

        risk_rf = get_risk_level(p_rf)
        risk_xgb = get_risk_level(p_xgb)

        # Combine predictions
        agreement = "Yes" if risk_rf == risk_xgb else "No"
        
        if agreement == "Yes":
            final_risk = risk_rf
            # Average probability for confidence
            p_combined = 0.5 * (p_rf + p_xgb)
            if final_risk == 'Low':
                confidence = (1.0 - p_combined) * 100.0
            else:
                confidence = p_combined * 100.0
        else:
            # Average probability decides
            p_combined = 0.5 * (p_rf + p_xgb)
            final_risk = get_risk_level(p_combined)
            if final_risk == 'Low':
                confidence = (1.0 - p_combined) * 100.0
            else:
                confidence = p_combined * 100.0
                
        # Confidence logic capping
        confidence = min(max(confidence, 55.0), 99.0)

        # Compile reasons for high/med risk
        debt_reasons = []
        if revol_util > 0.50:
            debt_reasons.append("High Credit Utilization: Using over 50% of revolving credit limits.")
        if emergency_ratio < 0.10:
            debt_reasons.append("Low Savings: Savings rate is below 10% of monthly earnings.")
        # High spending only if a significant portion of income has gone out
        outflow_ratio = total_expenses_30 / monthly_income

        if total_expenses_30 >= 2000 and outflow_ratio > 0.40:
            debt_reasons.append(
                f"High Monthly Outflow: ₹{total_expenses_30:.0f} spent in the last 30 days "
                f"({outflow_ratio*100:.1f}% of monthly income)."
            )
        if low_emergency_fund:
            debt_reasons.append(
                "Emergency Fund: Cash reserves are below the recommended 3-month safety buffer."
            )
        if debt_ratio > 0.40:
            debt_reasons.append("High Debt Ratio: Total debt burden exceeds 40% of annual income.")
            
        if not debt_reasons:
            debt_reasons.append("Healthy utilization of credit, low debt burden, and steady savings.")

        debt_explanation = " ".join(debt_reasons)
        
        #Recommendations for debt card
        # Recommendations for debt card
        if final_risk == "Low":
            if low_emergency_fund:
                debt_rec = (
                    "Debt risk is currently low. Continue your financial discipline while "
                    "building an emergency fund covering at least 3 months of income."
                )
            else:
                debt_rec = "Maintain your current financial discipline."

        elif final_risk == "Medium":
            debt_rec = (
                "Focus on building your emergency fund and avoid taking new personal "
                "or credit card debt."
            )

        else:  # High Risk
            debt_rec = (
                "Immediately reduce discretionary spending, lower outstanding debt, "
                "and consider consolidating high-interest loans."
            )
        # ============================================================
# COMBINE RF + XGBOOST SHAP VALUES
# ============================================================

        debt_shap = []

        if debt_shap_rf and debt_shap_xgb:

            xgb_lookup = {
                item['feature']: item['shap_value']
                for item in debt_shap_xgb
            }

            for item in debt_shap_rf:

                feature = item['feature']

                rf_value = item['shap_value']
                xgb_value = xgb_lookup.get(feature, 0.0)

                combined_value = (
                    (rf_value + xgb_value) / 2
                )

                debt_shap.append({
                    'feature': feature,
                    'value': item['value'],
                    'shap_value': round(combined_value, 4),
                    'absolute_shap': round(abs(combined_value), 4),
                    'impact': (
                        'Increases Risk'
                        if combined_value > 0
                        else 'Decreases Risk'
                        if combined_value < 0
                        else 'No Significant Impact'
                    )
                })

        debt_shap.sort(
            key=lambda x: x['absolute_shap'],
            reverse=True
        )

        debt_shap = debt_shap[:6]
        debt_data = {
            'score': p_combined * 100.0,
            'recommendation': debt_rec,
            'prediction': final_risk,
            'confidence': float(round(confidence, 1)),
            'explanation': debt_explanation,
            'feature_importance': {
                'Model Agreement': agreement,
                'RF Probability': round(p_rf * 100.0, 1),
                'XGBoost Probability': round(p_xgb * 100.0, 1),
                'Credit Utilization': round(revol_util * 100.0, 1),
                'Annual Debt Ratio': round(ml_debt_ratio * 100.0, 1),
            },
            'shap_explanation': debt_shap
        }
        print("credit_card_debt =", credit_card_debt)
        print("total_debt =", total_debt)
        print("monthly_income =", monthly_income)
        print("revol_util =", revol_util)
        print("ml_debt_ratio =", ml_debt_ratio)
        
        
        FinancialWellnessService.save_insight_db(user, 'Debt Risk', debt_data)

        # ---------------------------------------------------------------------
        # STEP 4: Savings Forecast (Linear Regression)
        # ---------------------------------------------------------------------
        # Check if the user has enough history (e.g. >= 3 months of database transactions)
        # Let's count user's months
        user_months_df = pd.DataFrame(list(all_tx.values('created_at', 'amount', 'transaction_type')))
        enough_user_data = False
        
        if not user_months_df.empty:
            user_months_df['Month'] = pd.to_datetime(user_months_df['created_at']).dt.to_period('M')
            unique_months = user_months_df['Month'].nunique()
            if unique_months >= 3:
                enough_user_data = True

        projected_3 = wallet_balance
        projected_6 = wallet_balance
        projected_12 = wallet_balance
        savings_explanation = ""
        # Initialize hybrid variables (populated in the fallback else-branch)
        current_total_savings = existing_savings
        actual_monthly_saving = 0.0
        ai_monthly_saving = 0.0
        hybrid_monthly_saving = 0.0

        if enough_user_data:
            print("Training user-specific Linear Regression for savings forecast...")
            monthly_user = []
            for month, group in user_months_df.groupby('Month'):
                credits = group[group['transaction_type'] == 'Deposit']['amount'].sum()
                # withdrawals + transfers
                debits = group[group['transaction_type'] != 'Deposit']['amount'].sum()
                monthly_user.append({
                    'Credits': float(credits),
                    'Debits': float(debits),
                    'Net': float(credits - debits)
                })
            df_u_monthly = pd.DataFrame(monthly_user)
            df_u_monthly['Cumulative'] = df_u_monthly['Net'].cumsum() + wallet_balance
            
            X_u = np.arange(len(df_u_monthly)).reshape(-1, 1)
            y_u = df_u_monthly['Cumulative'].values
            
            lr_u = LinearRegression()
            lr_u.fit(X_u, y_u)
            
            # Predict 3, 6, 12 months ahead
            last_idx = len(df_u_monthly) - 1
            projected_3 = float(lr_u.predict([[last_idx + 3]])[0])
            projected_6 = float(lr_u.predict([[last_idx + 6]])[0])
            projected_12 = float(lr_u.predict([[last_idx + 12]])[0])
            
            user_savings_rate = float(lr_u.coef_[0])
            savings_explanation = f"Based on your actual monthly net savings of ₹{round(user_savings_rate, 2)} over the past {unique_months} months, we project your savings trajectory."
        else:
            print("Using Hybrid AI + Actual Transaction History for Savings Forecast...")

            # ------------------------------------------------------------------
            # STEP 1 – Current Total Savings
            # Wallet Balance + Active FD Amount + Actual RD Installments Paid
            # + Current Passion Fund Amount  (no double-counting)
            # ------------------------------------------------------------------
            actual_rd_paid = float(
                Transaction.objects.filter(
                    user=user,
                    transaction_type="RD Installment"
                ).aggregate(total=Sum("amount"))["total"] or 0
            )

            current_total_savings = (
                wallet_balance
                + fd_total
                + actual_rd_paid
                + passion_total
            )

            # ------------------------------------------------------------------
            # STEP 2 – Actual Monthly Savings Behaviour (Behaviour-Based)
            # Combines planned saving intent with historical net transaction flow.
            #
            # Component A – Planned Monthly Savings (monthly_saved):
            #   monthly_savings_target + RD monthly amount + Passion Fund monthly amount
            #   Represents the customer's intended recurring saving behaviour.
            #
            # Component B – Average Monthly Net Savings (transaction history):
            #   Average of (Deposits - Withdrawals - Transfers) per month.
            #   Used ONLY as a behavioural adjustment, NOT as the primary figure,
            #   to prevent one-time large deposits from inflating the forecast.
            #
            # Formula:
            #   behaviour_monthly_saving = (0.70 × monthly_saved)
            #                             + (0.30 × average_monthly_net_savings)
            #   actual_monthly_saving    = behaviour_monthly_saving
            # ------------------------------------------------------------------
            all_tx_df = pd.DataFrame(
                list(
                    Transaction.objects.filter(user=user)
                    .values("created_at", "amount", "transaction_type")
                )
            )

            if not all_tx_df.empty:
                all_tx_df["Month"] = pd.to_datetime(all_tx_df["created_at"]).dt.to_period("M")
                monthly_nets = []
                for _month, grp in all_tx_df.groupby("Month"):
                    m_deposits = float(
                        grp[grp["transaction_type"] == "Deposit"]["amount"].sum()
                    )
                    m_withdrawals = float(
                        grp[grp["transaction_type"] == "Withdraw"]["amount"].sum()
                    )
                    m_transfers = float(
                        grp[grp["transaction_type"] == "Transfer"]["amount"].sum()
                    )
                    monthly_nets.append(m_deposits - m_withdrawals - m_transfers)
                average_monthly_net_savings = float(np.mean(monthly_nets)) if monthly_nets else 0.0
            else:
                average_monthly_net_savings = 0.0

            # Behaviour-based blend: planned saving intent (70%) + net transaction evidence (30%)
            behaviour_monthly_saving = (0.70 * monthly_saved) + (0.30 * average_monthly_net_savings)
            actual_monthly_saving = behaviour_monthly_saving

            # ------------------------------------------------------------------
            # STEP 3 – AI Model Prediction (Linear Regression baseline)
            # Load savings_baselines.pkl; scale base_savings_rate by income.
            # ------------------------------------------------------------------
            lr_baselines = get_saved_model("savings_baselines.pkl")

            if lr_baselines is not None:
                base_savings_rate = lr_baselines["avg_monthly_net_savings"]
                base_income = lr_baselines["avg_monthly_income"]
                ai_monthly_saving = base_savings_rate * (
                    monthly_income / max(base_income, 1)
                )
            else:
                # Fallback: 10% of income as AI estimate
                ai_monthly_saving = monthly_income * 0.10

            # ------------------------------------------------------------------
            # STEP 3b – AI Calibration
            # If the AI estimate is significantly higher than the customer's
            # actual monthly saving behaviour, cap it at 1.5× the actual rate.
            # This keeps the AI optimistic without producing unrealistic forecasts.
            # ------------------------------------------------------------------
            if actual_monthly_saving > 0 and ai_monthly_saving > actual_monthly_saving * 1.5:
                ai_monthly_saving = actual_monthly_saving * 1.5

            # ------------------------------------------------------------------
            # STEP 4 – Hybrid Monthly Saving Rate
            # 70% Actual Banking Behaviour + 30% AI Prediction
            # ------------------------------------------------------------------
            hybrid_monthly_saving = (
                actual_monthly_saving * 0.70
                + ai_monthly_saving * 0.30
            )

            # ------------------------------------------------------------------
            # DEBUG – Savings Forecast Values (remove or comment out in production)
            # ------------------------------------------------------------------
            print("[Savings Forecast DEBUG]")
            print(f"  Actual Monthly Saving  : ₹{round(actual_monthly_saving, 2)}")
            print(f"  AI Monthly Saving      : ₹{round(ai_monthly_saving, 2)}")
            print(f"  Hybrid Monthly Saving  : ₹{round(hybrid_monthly_saving, 2)}")

            # ------------------------------------------------------------------
            # STEP 5 – Project Future Savings
            # Base: current_total_savings (NOT just wallet balance)
            # ------------------------------------------------------------------
            projected_3  = round(current_total_savings + hybrid_monthly_saving * 3,  2)
            projected_6  = round(current_total_savings + hybrid_monthly_saving * 6,  2)
            projected_12 = round(current_total_savings + hybrid_monthly_saving * 12, 2)

            print(f"  Projected 12-Month Savings: ₹{projected_12}")

            user_savings_rate = hybrid_monthly_saving

            # ------------------------------------------------------------------
            # STEP 6 – Explanation
            # ------------------------------------------------------------------
            savings_explanation = (
                f"This forecast combines your actual banking behaviour (70%) with our "
                f"AI prediction model (30%). Current savings of "
                f"₹{round(current_total_savings, 2)} and an estimated monthly saving "
                f"rate of ₹{round(hybrid_monthly_saving, 2)} were used to forecast "
                f"future savings."
            )

        # Keep values non-negative
        projected_3  = max(0.0, round(projected_3,  2))
        projected_6  = max(0.0, round(projected_6,  2))
        projected_12 = max(0.0, round(projected_12, 2))

        savings_rec = "Increase savings by ₹2000/month to speed up wealth creation."
        if user_savings_rate > monthly_savings_target:
            savings_rec = "Maintain current spending. Consider locking in surplus into a Fixed Deposit for higher returns."

        # ------------------------------------------------------------------
        # STEP 7 – Feature Importance (7 required keys)
        # ------------------------------------------------------------------
        # Determine whether we are in the enough_user_data branch or the
        # fallback branch so we can populate hybrid/AI keys meaningfully.
        _in_fallback = not enough_user_data

        savings_data = {
            'score': float(projected_12),
            'recommendation': savings_rec,
            'prediction': f"₹{projected_12}",
            'confidence': 85.0,
            'explanation': savings_explanation,
            'feature_importance': {
                'Current Savings':        round(current_total_savings if _in_fallback else existing_savings, 2),
                'Actual Monthly Saving':  round(actual_monthly_saving  if _in_fallback else user_savings_rate, 2),
                'AI Monthly Saving':      round(ai_monthly_saving       if _in_fallback else user_savings_rate, 2),
                'Hybrid Monthly Saving':  round(hybrid_monthly_saving   if _in_fallback else user_savings_rate, 2),
                # Keys used by the HTML
                'forecast_3_months': projected_3,
                'forecast_6_months': projected_6,
                'forecast_12_months': projected_12,
                'monthly_savings_rate': round(
                    hybrid_monthly_saving if _in_fallback else user_savings_rate,
                    2
    ),
            }
        }
        FinancialWellnessService.save_insight_db(user, 'Savings Forecast', savings_data)
        

        # ---------------------------------------------------------------------
        # STEP 5: Spending Pattern Detection (K-Means Clustering - Rolling 90 Days)
        # ---------------------------------------------------------------------
        scaler = get_saved_model('spending_scaler.pkl')
        kmeans = get_saved_model('spending_kmeans.pkl')
        cluster_meta = get_saved_model('spending_clusters_metadata.pkl')
        
        standard_categories = ['Food', 'Shopping', 'Travel', 'Bills', 'Healthcare', 'Education', 'Investment', 'Subscription', 'Other']
        user_cat_spend = {cat: 0.0 for cat in standard_categories}
        
        # Retrieve user's debit transactions in the last 90 days
        last_90_days = now - timedelta(days=90)
        user_debits_90 = Transaction.objects.filter(
            user=user,
            created_at__gte=last_90_days,
            transaction_type__in=['Withdraw', 'Transfer']
        )
        total_user_debit_90 = float(user_debits_90.aggregate(Sum('amount'))['amount__sum'] or 0.0)

        # Retrieve user's debit transactions in the last 30 days (for month-over-month comparison)
        user_debits_30 = recent_tx.filter(transaction_type__in=['Withdraw', 'Transfer'])
        total_user_debit = float(user_debits_30.aggregate(Sum('amount'))['amount__sum'] or 0.0)
        
        # Categorize
        category_map = {
            "Food": "Food",
            "Shopping": "Shopping",
            "Bills": "Bills",
            "Travel": "Travel",
            "Healthcare": "Healthcare",
            "Education": "Education",
            "Investment": "Investment",
            "Subscription": "Subscription",
            "Entertainment": "Shopping",
            "Rent": "Bills",
            "Family": "Other",
            "Other": "Other",
        }
        
        for tx in user_debits_90:
            tx_cat = tx.category
            mapped = category_map.get(tx_cat, 'Other')
            user_cat_spend[mapped] += float(tx.amount)

        # Normalize to percentages based on 90-day debits
        user_percentages = {cat: 0.0 for cat in standard_categories}
        if total_user_debit_90 > 0:
            for cat in standard_categories:
                user_percentages[cat] = (user_cat_spend[cat] / total_user_debit_90) * 100.0
        else:
            # Default mock percentages matching typical checking account
            user_percentages = {
                'Food': 25.0, 'Shopping': 20.0, 'Travel': 10.0, 'Bills': 35.0, 
                'Healthcare': 5.0, 'Education': 0.0, 'Investment': 0.0, 
                'Subscription': 5.0, 'Other': 0.0
            }

        # Predict cluster
        cluster_id = 0
        if scaler is not None and kmeans is not None:
            try:
                user_vec = np.array([[user_percentages[cat] for cat in standard_categories]])
                user_vec_scaled = scaler.transform(user_vec)
                cluster_id = int(kmeans.predict(user_vec_scaled)[0])
            except Exception as e:
                print(f"Error predicting spending cluster: {e}")
                cluster_id = 0

        # Description of spending profiles
        profile_names = {
            0: "Bills & Housing Dominated",
            1: "Discretionary (Shopping & Dining) Heavy",
            2: "Balanced & Lifestyle Spending"
        }
        
        pred_pattern = profile_names.get(cluster_id, f"Pattern {cluster_id+1}")
        
        # Identify top spending category based on 90-day data
        top_category = max(user_percentages, key=user_percentages.get)
        if total_user_debit_90 <= 0:
            top_category = "N/A"

        # Determine monthly trend
        # Retrieve previous 30 days expenses (30 to 60 days ago)
        prev_start = now - timedelta(days=60)
        prev_end = now - timedelta(days=30)
        prev_tx = Transaction.objects.filter(user=user, created_at__gte=prev_start, created_at__lt=prev_end)
        prev_expenses = float(prev_tx.filter(transaction_type__in=['Withdraw', 'Transfer']).aggregate(Sum('amount'))['amount__sum'] or 0.0)

        increasing_expenses = "None"
        decreasing_expenses = "None"
        
        if prev_expenses > 0 and total_user_debit > 0:
            percent_change = ((total_user_debit - prev_expenses) / prev_expenses) * 100.0
            if percent_change > 5.0:
                monthly_trend = f"Expenses increased by {round(percent_change, 1)}% from last month"
                increasing_expenses = top_category
            elif percent_change < -5.0:
                monthly_trend = f"Expenses decreased by {round(abs(percent_change), 1)}% from last month"
                decreasing_expenses = top_category
            else:
                monthly_trend = "Spending is stable compared to last month"
        else:
            monthly_trend = "Spending trend is stable (insufficient historical months)"

        # Determine weekly trend (compare last 7 days vs previous 7 days)
        last_7_days = now - timedelta(days=7)
        prev_7_days = now - timedelta(days=14)
        week1_spend = float(recent_tx.filter(created_at__gte=last_7_days, transaction_type__in=['Withdraw', 'Transfer']).aggregate(Sum('amount'))['amount__sum'] or 0.0)
        week2_spend = float(recent_tx.filter(created_at__gte=prev_7_days, created_at__lt=last_7_days, transaction_type__in=['Withdraw', 'Transfer']).aggregate(Sum('amount'))['amount__sum'] or 0.0)

        if week2_spend > 0:
            w_change = ((week1_spend - week2_spend) / week2_spend) * 100.0
            if w_change > 10.0:
                weekly_trend = f"Weekly spending increased by {round(w_change, 1)}%"
            elif w_change < -10.0:
                weekly_trend = f"Weekly spending decreased by {round(abs(w_change), 1)}%"
            else:
                weekly_trend = "Weekly spending is stable"
        else:
            weekly_trend = "Weekly spending is stable"

        if total_user_debit_90 <= 0:
            spending_explanation = f"No debit transactions recorded in the last 90 days. Baseline '{pred_pattern}' profile assigned. {monthly_trend}."
        else:
            spending_explanation = f"Your spending matches the '{pred_pattern}' profile based on your last 90 days of activity. Top category is {top_category} (accounting for {round(user_percentages.get(top_category, 0), 1)}% of total 90-day debit outflows). {monthly_trend}."

        spending_rec = "Reduce shopping expenses to release savings buffer."
        if top_category == 'Food' and user_percentages['Food'] > 30:
            spending_rec = "Reduce restaurant dine-outs and track grocery spending."
        elif top_category == 'Bills':
            spending_rec = "Audit recurring bills and utility plans to save cash."
        elif top_category == 'Shopping':
            spending_rec = "Reduce shopping expenses and defer discretionary purchases."
        else:
            spending_rec = "Maintain current spending pattern."

        # Features format
        spending_feat_imp = {
            'Top Spending Category': top_category,
            'Monthly Trend': monthly_trend,
            'Weekly Trend': weekly_trend,
            'Increasing Expenses': increasing_expenses,
            'Decreasing Expenses': decreasing_expenses,
            'Cluster ID': cluster_id,
            'Food Percentage': round(user_percentages['Food'], 1),
            'Shopping Percentage': round(user_percentages['Shopping'], 1),
            'Bills Percentage': round(user_percentages['Bills'], 1),
            'Travel Percentage': round(user_percentages['Travel'], 1),
            'Subscription Percentage': round(user_percentages['Subscription'], 1),
            '90-Day Spending': round(total_user_debit_90,2),
            '30-Day Spending': round(total_user_debit,2),
            'Previous Month Spending': round(prev_expenses,2),
        }

        spending_data = {
            'score': float(cluster_id),
            'recommendation': spending_rec,
            'prediction': pred_pattern,
            'confidence': 90.0,
            'explanation': spending_explanation,
            'feature_importance': spending_feat_imp
        }
        FinancialWellnessService.save_insight_db(user, 'Impulse Spending', spending_data, 'Spending Pattern')

        # ---------------------------------------------------------------------
        # STEP 6: Poverty Risk Analysis (RULE-BASED — CUMULATIVE WEIGHTED SCORE)
        # ---------------------------------------------------------------------
        # Inputs used:
        #   Monthly Income, Wallet Balance, FD Balance, RD Balance, Passion Fund Balance,
        #   Monthly Withdrawals, Monthly Transfers, Monthly Savings Target,
        #   Monthly RD Installments, Monthly Passion Fund Installments,
        #   Total Outstanding Debt, Preferred Monthly Budget,
        #   Employment Type, Dependents
        #
        # Each vulnerability condition adds independent points to a 0–100 risk score.
        # Classification:
        #   0–29  → Low Risk
        #   30–59 → Medium Risk
        #   60+   → High Risk
        #
        # This avoids the fragile priority-ordering problem of nested if-else chains.
        # Every contributing factor is independently traceable in feature_importance.
        # ---------------------------------------------------------------------

        # ── Derived metrics ──────────────────────────────────────────────────
        pov_total_savings   = wallet_balance + fd_total + rd_total + passion_total
        pov_monthly_expenses = total_withdrawals_30 + total_transfers_30
        pov_monthly_planned_savings = (
            monthly_savings_target + total_rd_monthly + total_passion_monthly
        )

        # Emergency fund: how many months of income are covered by total savings
        pov_emergency_months = pov_total_savings / monthly_income  # monthly_income >= 1 guaranteed

        # Debt ratio: annual debt vs annual income
        pov_debt_ratio = total_debt / (monthly_income * 12.0) if monthly_income > 0 else 0.0
        print("pov_debt_ratio =", pov_debt_ratio)

        # Savings rate: planned monthly savings as % of monthly income
        pov_savings_rate = pov_monthly_planned_savings / monthly_income if monthly_income > 0 else 0.0

        # Budget utilisation: actual expenses vs preferred budget
        pov_budget_util = (
            pov_monthly_expenses / preferred_monthly_budget
            if preferred_monthly_budget > 0
            else (pov_monthly_expenses / monthly_income if monthly_income > 0 else 0.0)
        )

        # ── Cumulative risk scoring ───────────────────────────────────────────
        # Each condition below adds points independently; higher total = higher risk.
        pov_risk_score   = 0
        pov_risk_factors = {}   # label → points contributed (for feature_importance)
        pov_reasons      = []   # human-readable explanation bullets

        # 1. Income level (max 20 pts)
        if monthly_income < 15_000:
            pts = 20
            pov_risk_score += pts
            pov_risk_factors['Income Level'] = pts
            pov_reasons.append(
                f"Very low monthly income (₹{round(monthly_income, 0):,.0f}) leaves little margin to absorb financial shocks."
            )
        elif monthly_income < 30_000:
            pts = 12
            pov_risk_score += pts
            pov_risk_factors['Income Level'] = pts
            pov_reasons.append(
                f"Moderate-low income (₹{round(monthly_income, 0):,.0f}) limits financial flexibility."
            )
        elif monthly_income < 50_000:
            pts = 5
            pov_risk_score += pts
            pov_risk_factors['Income Level'] = pts
        else:
            pov_risk_factors['Income Level'] = 0

        # 2. Emergency fund depth (max 20 pts)
        if pov_emergency_months < 1.0:
            pts = 20
            pov_risk_score += pts
            pov_risk_factors['Emergency Fund'] = pts
            pov_reasons.append(
                f"Emergency savings cover less than 1 month of income "
                f"({round(pov_emergency_months, 2)} months) — critically low."
            )
        elif pov_emergency_months < 2.0:
            pts = 12
            pov_risk_score += pts
            pov_risk_factors['Emergency Fund'] = pts
            pov_reasons.append(
                f"Emergency fund covers {round(pov_emergency_months, 2)} months "
                f"— below the recommended 3-month buffer."
            )
        elif pov_emergency_months < 3.0:
            pts = 5
            pov_risk_score += pts
            pov_risk_factors['Emergency Fund'] = pts
            pov_reasons.append(
                f"Emergency fund covers {round(pov_emergency_months, 2)} months "
                f"— approaching but not yet at the 3-month safety margin."
            )
        else:
            pov_risk_factors['Emergency Fund'] = 0

        # 3. Savings rate (max 15 pts)
        if pov_savings_rate < 0.05:
            pts = 15
            pov_risk_score += pts
            pov_risk_factors['Savings Rate'] = pts
            pov_reasons.append(
                f"Planned monthly savings rate is very low "
                f"({round(pov_savings_rate * 100, 1)}% of income)."
            )
        elif pov_savings_rate < 0.10:
            pts = 8
            pov_risk_score += pts
            pov_risk_factors['Savings Rate'] = pts
            pov_reasons.append(
                f"Savings rate of {round(pov_savings_rate * 100, 1)}% is below "
                f"the recommended 10% minimum."
            )
        elif pov_savings_rate < 0.20:
            pts = 3
            pov_risk_score += pts
            pov_risk_factors['Savings Rate'] = pts
        else:
            pov_risk_factors['Savings Rate'] = 0

        # 4. Debt burden (max 15 pts)
        if pov_debt_ratio > 0.50:
            pts = 15
            pov_risk_score += pts
            pov_risk_factors['Debt Ratio'] = pts
            pov_reasons.append(
                f"Outstanding debt exceeds 50% of annual income "
                f"(ratio: {round(pov_debt_ratio, 3)}) — severely burdens cash flow."
            )
        elif pov_debt_ratio > 0.35:
            pts = 10
            pov_risk_score += pts
            pov_risk_factors['Debt Ratio'] = pts
            pov_reasons.append(
                f"High debt burden (ratio: {round(pov_debt_ratio, 3)}) "
                f"— exceeds the 35% healthy ceiling."
            )
        elif pov_debt_ratio > 0.20:
            pts = 4
            pov_risk_score += pts
            pov_risk_factors['Debt Ratio'] = pts
        else:
            pov_risk_factors['Debt Ratio'] = 0

        # 5. Overspending / budget utilisation (max 15 pts)
        if pov_budget_util > 1.30:
            pts = 15
            pov_risk_score += pts
            pov_risk_factors['Budget Utilisation'] = pts
            pov_reasons.append(
                f"Monthly expenses exceed the budget by "
                f"{round((pov_budget_util - 1) * 100, 1)}% — significant overspending detected."
            )
        elif pov_budget_util > 1.10:
            pts = 8
            pov_risk_score += pts
            pov_risk_factors['Budget Utilisation'] = pts
            pov_reasons.append(
                f"Spending is {round((pov_budget_util - 1) * 100, 1)}% above budget "
                f"— minor overspending detected."
            )
        elif pov_budget_util > 0.90:
            pts = 3
            pov_risk_score += pts
            pov_risk_factors['Budget Utilisation'] = pts
        else:
            pov_risk_factors['Budget Utilisation'] = 0

        # 6. Dependents burden (max 10 pts)
        if dependents >= 4:
            pts = 10
            pov_risk_score += pts
            pov_risk_factors['Dependents'] = pts
            pov_reasons.append(
                f"{dependents} dependents create a high household financial burden."
            )
        elif dependents >= 3:
            pts = 6
            pov_risk_score += pts
            pov_risk_factors['Dependents'] = pts
            pov_reasons.append(
                f"{dependents} dependents add moderate financial pressure to the household."
            )
        elif dependents >= 1:
            pts = 2
            pov_risk_score += pts
            pov_risk_factors['Dependents'] = pts
        else:
            pov_risk_factors['Dependents'] = 0

        # 7. Employment / income stability (max 5 pts)
        unstable_types = ['Student', 'Freelancer', 'Self-Employed', 'Other']
        if employment_type in unstable_types or monthly_income <= 100.0:
            pts = 5
            pov_risk_score += pts
            pov_risk_factors['Employment Stability'] = pts
            pov_reasons.append(
                f"Employment type '{employment_type}' carries variable or uncertain income risk."
            )
        else:
            pov_risk_factors['Employment Stability'] = 0

        # ── Classification ────────────────────────────────────────────────────
        pov_risk_score = min(pov_risk_score, 100)   # cap at 100

        if pov_risk_score >= 60:
            poverty_risk_level = 'High'
        elif pov_risk_score >= 30:
            poverty_risk_level = 'Medium'
        else:
            poverty_risk_level = 'Low'

        # ── Explanation ───────────────────────────────────────────────────────
        if pov_reasons:
            poverty_explanation = (
                f"Financial vulnerability score: {pov_risk_score}/100 ({poverty_risk_level} Risk). "
                + " ".join(pov_reasons)
            )
        else:
            poverty_explanation = (
                f"Financial vulnerability score: {pov_risk_score}/100. "
                "No significant vulnerability indicators detected. "
                "Maintain current savings and spending discipline."
            )

        # ── Recommendation ────────────────────────────────────────────────────
        if poverty_risk_level == 'High':
            poverty_rec = (
                "Immediately prioritise building a cash reserve of at least 1 month of income. "
                "Clear high-interest credit card balances first, reduce discretionary spending, "
                "and explore income augmentation or assistance programmes."
            )
        elif poverty_risk_level == 'Medium':
            poverty_rec = (
                "Automate a recurring deposit of ₹1,000–₹2,000 per month to grow your emergency buffer. "
                "Review and trim any overspending categories and aim for a 10%+ savings rate."
            )
        else:
            poverty_rec = (
                "Maintain current savings discipline. Consider locking surplus funds into a "
                "Fixed Deposit or Passion Fund to grow wealth and further strengthen your safety net."
            )

        poverty_data = {
            'score': float(pov_risk_score),
            'recommendation': poverty_rec,
            'prediction': poverty_risk_level,
            'confidence': 100.0,   # Rule-based → fully deterministic
            'explanation': poverty_explanation,
            'feature_importance': {

    # ---------- Profile ----------
    'Monthly Income': round(monthly_income, 2),
    'Employment Type': employment_type,
    'Dependents': dependents,

    # ---------- Savings ----------
    'Wallet Balance': round(wallet_balance, 2),
    'Fixed Deposit Balance': round(fd_total, 2),
    'Recurring Deposit Balance': round(rd_total, 2),
    'Passion Fund Balance': round(passion_total, 2),
    'Total Savings': round(pov_total_savings, 2),

    # ---------- Spending ----------
    'Monthly Withdrawals': round(total_withdrawals_30, 2),
    'Monthly Transfers': round(total_transfers_30, 2),
    'Monthly Expenses': round(pov_monthly_expenses, 2),

    # ---------- Saving Behaviour ----------
    'Savings Target': round(monthly_savings_target,2),
    'RD Monthly Installment': round(total_rd_monthly,2),
    'Passion Fund Monthly Installment': round(total_passion_monthly,2),

    'Savings Rate (%)': round(pov_savings_rate*100,2),

    # ---------- Debt ----------
    'Home Loan': round(home_loan,2),
    'Education Loan': round(education_loan,2),
    'Personal Loan': round(personal_loan,2),
    'Credit Card Debt': round(credit_card_debt,2),
    'Other Debt': round(other_debt,2),
    'Total Debt': round(total_debt,2),

    # ---------- Budget ----------
    'Preferred Budget': round(preferred_monthly_budget,2),
    'Budget Utilisation': round(pov_budget_util,3),

    # ---------- Emergency ----------
    'Emergency Months': round(pov_emergency_months,2),

    # ---------- Final Score ----------
    'Risk Score': pov_risk_score,

    'Pts – Income': pov_risk_factors.get('Income Level',0),
    'Pts – Emergency Fund': pov_risk_factors.get('Emergency Fund',0),
    'Pts – Savings Rate': pov_risk_factors.get('Savings Rate',0),
    'Pts – Debt': pov_risk_factors.get('Debt Ratio',0),
    'Pts – Overspending': pov_risk_factors.get('Budget Utilisation',0),
    'Pts – Dependents': pov_risk_factors.get('Dependents',0),
    'Pts – Employment': pov_risk_factors.get('Employment Stability',0),
}
        }
        FinancialWellnessService.save_insight_db(user, 'Poverty Risk', poverty_data)

        # ---------------------------------------------------------------------
        # STEP 7: AI Recommendation Engine (Consolidate recommendations)
        # ---------------------------------------------------------------------
        # We consolidate recommendation triggers into a custom result structure
        ai_recommendations = []
        if emergency_ratio < 0.15:
            ai_recommendations.append("Increase savings rate: Allocate at least 15% of your income to savings.")
        if low_emergency_fund:
            ai_recommendations.append("Build emergency fund: Save up to 3 months of income in your wallet.")
        if expense_ratio > 0.55:
            ai_recommendations.append("Control discretionary spending: Reduce shopping and dining-out expenditures.")
        if total_debt > 0 and debt_ratio > 0.35:
            ai_recommendations.append("Reduce debt ratio: Focus on clearing outstanding credit card debts.")
        if poverty_risk_level == 'High':
            ai_recommendations.append("Safety buffer: Restructure loans and minimize financial outlays immediately.")
        
        if not ai_recommendations:
            ai_recommendations.append("Excellent management: Maintain current balanced spending and saving pattern.")

        recommendation_block = {
            'Financial Wellness': wellness_data,
            'Debt Risk': debt_data,
            'Savings Forecast': savings_data,
            'Impulse Spending': spending_data,
            'Poverty Risk': poverty_data,
            'Consolidated Recommendations': ai_recommendations
        }

        print(f"Recalculation complete for {user.customer_id}.")
        return recommendation_block

    @staticmethod
    def save_insight_db(user, insight_type, data, model_name=None):
        """Save AI insight and explanation including SHAP data."""

        insight = AIInsight.objects.create(
            user=user,
            insight_type=insight_type,
            score=data['score'],
            recommendation=data['recommendation']
        )

        # Copy feature importance so we don't modify the original dictionary
        feature_importance = dict(
            data.get('feature_importance', {})
        )

        # Store SHAP inside feature_importance for database persistence
        shap_explanation = data.get(
            'shap_explanation',
            []
        )

        feature_importance['__shap__'] = shap_explanation

        AIExplanation.objects.create(
            user=user,
            model_name=model_name if model_name else insight_type,
            prediction=data['prediction'],
            confidence=data['confidence'],
            explanation=data['explanation'],
            feature_importance=feature_importance
        )

    @staticmethod
    def generate_charts(user):
        """
        Generates Plotly interactive chart HTML strings for:
        1. Savings Growth Trend (Scatter mode='lines+markers')
        2. Expense Distribution (Pie hole=0.5)
        Using a 90-day rolling transaction window.
        """
        now = timezone.now()
        last_90_days = now - timedelta(days=90)

        # ------------------------------------------------------------------
        # 1. Savings Projection Growth Line Chart
        # ------------------------------------------------------------------
        tx_90 = Transaction.objects.filter(user=user, created_at__gte=last_90_days).order_by('created_at')

        if tx_90.exists():
            df_tx = pd.DataFrame(list(tx_90.values('created_at', 'amount', 'transaction_type')))
            df_tx['created_at'] = pd.to_datetime(df_tx['created_at'])
            df_tx['created_at'] = df_tx['created_at'].dt.tz_localize(None)
            df_tx['Month_Period'] = df_tx['created_at'].dt.to_period('M')
            df_tx['Month_Name'] = df_tx['created_at'].dt.strftime('%b %Y')

            months = []
            cumulative_savings = []
            running_sum = 0.0

            grouped = df_tx.groupby(['Month_Period', 'Month_Name'], sort=True)

            inflow_types = ['Deposit', 'RD Installment', 'Passion Fund Topup', 'Passion Fund']
            outflow_types = ['Withdraw', 'Transfer']

            for (period, m_name), grp in grouped:
                inflows = float(grp[grp['transaction_type'].isin(inflow_types)]['amount'].sum())
                outflows = float(grp[grp['transaction_type'].isin(outflow_types)]['amount'].sum())
                net_savings = inflows - outflows
                running_sum += net_savings
                months.append(m_name)
                cumulative_savings.append(round(running_sum, 2))

            fig_sav = go.Figure()
            fig_sav.add_trace(go.Scatter(
                x=months,
                y=cumulative_savings,
                mode='lines+markers',
                line=dict(color='#10B981', width=3),
                marker=dict(size=8, color='#0B1026', line=dict(color='#10B981', width=2)),
                name='Cumulative Savings'
            ))
            fig_sav.update_layout(
                title='Savings Growth Trend',
                xaxis_title='Month',
                yaxis_title='Savings (₹)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="'Poppins', sans-serif", size=12, color="#64748B"),
                margin=dict(l=40, r=20, t=40, b=40),
                height=320,
                xaxis=dict(showgrid=True, gridcolor='#E2E8F0'),
                yaxis=dict(showgrid=True, gridcolor='#E2E8F0')
            )
            savings_growth_chart = fig_sav.to_html(full_html=False, include_plotlyjs=False)
        else:
            fig_sav = go.Figure()
            fig_sav.add_trace(go.Scatter(
                x=['Current Month'],
                y=[0.0],
                mode='lines+markers',
                line=dict(color='#10B981', width=3),
                marker=dict(size=8, color='#10B981')
            ))
            fig_sav.update_layout(
                title='Savings Growth Trend',
                xaxis_title='Month',
                yaxis_title='Savings (₹)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="'Poppins', sans-serif", size=12, color="#64748B"),
                margin=dict(l=40, r=20, t=40, b=40),
                height=320
            )
            savings_growth_chart = fig_sav.to_html(full_html=False, include_plotlyjs=False)

        # ------------------------------------------------------------------
        # 2. Monthly Expenditure Distribution Donut Chart
        # ------------------------------------------------------------------
        debit_tx_90 = Transaction.objects.filter(
            user=user,
            created_at__gte=last_90_days,
            transaction_type__in=['Withdraw', 'Transfer']
        )

        category_totals = {}
        for tx in debit_tx_90:
            cat = tx.category or 'Other'
            category_totals[cat] = category_totals.get(cat, 0.0) + float(tx.amount)

        if category_totals:
            labels = list(category_totals.keys())
            values = list(category_totals.values())

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=0.5,
                textinfo='label+percent',
                hovertemplate='<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>',
                marker=dict(colors=['#10B981', '#3B82F6', '#F59E0B', '#EF4444', '#8B5CF6', '#EC4899', '#6B7280', '#14B8A6', '#F97316'])
            )])
            fig_donut.update_layout(
                title='Expense Distribution',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="'Poppins', sans-serif", size=12, color="#64748B"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320,
                showlegend=True
            )
            expense_donut_chart = fig_donut.to_html(full_html=False, include_plotlyjs=False)
        else:
            fig_donut = go.Figure(data=[go.Pie(
                labels=['No Expense Data'],
                values=[100],
                hole=0.5,
                textinfo='label+percent'
            )])
            fig_donut.update_layout(
                title='Expense Distribution',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="'Poppins', sans-serif", size=12, color="#64748B"),
                margin=dict(l=20, r=20, t=40, b=20),
                height=320
            )
            expense_donut_chart = fig_donut.to_html(full_html=False, include_plotlyjs=False)

        return {
            'savings_growth_chart': savings_growth_chart,
            'expense_donut_chart': expense_donut_chart
        }
