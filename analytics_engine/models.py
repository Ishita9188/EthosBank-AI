from django.db import models
from accounts.models import UserAccount

# ===============================
# 1. Offer Analysis
# ===============================
class OfferAnalysis(models.Model):
    RISK_CHOICES = [
        ('Low', 'Low'),
        ('Medium', 'Medium'),
        ('High', 'High')
    ]

    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE
    )
    offer_text = models.TextField()
    manipulation_score = models.FloatField(default=0)
    transparency_score = models.FloatField(default=0)
    urgency_score = models.FloatField(default=0)
    sentiment_score = models.FloatField(default=0)
    risk_level = models.CharField(
        max_length=20,
        choices=RISK_CHOICES,
        default='Low'
    )
    recommendation = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.customer_id} - Offer Analysis"


# ===============================
# 2. Fraud Alerts
# ===============================
class FraudAlert(models.Model):
    STATUS_CHOICES = [
        ('Pending','Pending'),
        ('Resolved','Resolved'),
        ('Ignored','Ignored')
    ]
    ALERT_TYPES = [
        ('Suspicious Transfer','Suspicious Transfer'),
        ('Elder Fraud','Elder Fraud'),
        ('Phishing','Phishing'),
        ('Manipulative Offer','Manipulative Offer'),
        ('High Risk Transaction','High Risk Transaction')
    ]

    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE
    )
    alert_type = models.CharField(
        max_length=100,
        choices=ALERT_TYPES
    )
    alert_score = models.FloatField()
    description = models.TextField()
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='Pending'
    )
    generated_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(
        null=True,
        blank=True
    )

    def __str__(self):
        return f"{self.user.customer_id} - {self.alert_type}"


# ===============================
# 3. AI Insights
# ===============================
class AIInsight(models.Model):
    INSIGHT_TYPES = [
        ('Debt Risk','Debt Risk'),
        ('Savings Forecast','Savings Forecast'),
        ('Financial Wellness','Financial Wellness'),
        ('Impulse Spending','Impulse Spending'),
        ('Subscription Trap','Subscription Trap'),
        ('Poverty Risk','Poverty Risk')
    ]

    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE
    )
    insight_type = models.CharField(
        max_length=100,
        choices=INSIGHT_TYPES
    )
    score = models.FloatField()
    recommendation = models.TextField()
    generated_on = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.user.customer_id} - {self.insight_type}"


# ===============================
# 4. Beneficiaries
# ===============================
class Beneficiary(models.Model):
    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE
    )
    beneficiary_name = models.CharField(
        max_length=150
    )
    beneficiary_account = models.CharField(
        max_length=30
    )
    beneficiary_bank = models.CharField(
        max_length=100
    )
    ifsc_code = models.CharField(
        max_length=20
    )
    trusted = models.BooleanField(
        default=False
    )
    verified = models.BooleanField(
        default=False
    )
    added_on = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.beneficiary_name


# ===============================
# 5. OTP Verification
# ===============================
class OTPVerification(models.Model):
    PURPOSES = [
        ('Transfer','Transfer'),
        ('Fixed Deposit','Fixed Deposit'),
        ('Recurring Deposit','Recurring Deposit'),
        ('Password Reset','Password Reset'),
        ('Registration','Registration')
    ]

    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE
    )
    otp_code = models.CharField(
        max_length=6
    )
    purpose = models.CharField(
        max_length=50,
        choices=PURPOSES
    )
    verified = models.BooleanField(
        default=False
    )
    created_at = models.DateTimeField(
        auto_now_add=True
    )
    expires_at = models.DateTimeField()

    def __str__(self):
        return f"{self.user.customer_id} - {self.purpose}"


# ===============================
# 6. Explainable AI
# ===============================
class AIExplanation(models.Model):
    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE
    )
    model_name = models.CharField(
        max_length=100
    )
    prediction = models.CharField(
        max_length=200
    )
    confidence = models.FloatField(default=0)
    explanation = models.TextField()
    feature_importance = models.JSONField(
        blank=True,
        null=True
    )
    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.user.customer_id} - {self.model_name}"
