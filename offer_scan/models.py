from django.db import models
from accounts.models import UserAccount


class OfferScanHistory(models.Model):
    user = models.ForeignKey(UserAccount, on_delete=models.SET_NULL, null=True, blank=True)

    original_text = models.TextField()
    cleaned_text = models.TextField(blank=True)
    processed_text = models.TextField(blank=True)

    sentiment = models.CharField(max_length=100, blank=True)
    banking_intent = models.CharField(max_length=150, blank=True)
    ethical_classification = models.CharField(max_length=150, blank=True)
    ml_risk_prediction = models.CharField(max_length=100, blank=True)

    risk_level = models.CharField(max_length=50)
    risk_score = models.FloatField(default=0)
    manipulation_score = models.FloatField(default=0)
    transparency_score = models.FloatField(default=0)

    offer_category = models.CharField(max_length=100, blank=True)

    detected_patterns = models.JSONField(default=list, blank=True)
    pattern_analysis = models.JSONField(default=list, blank=True)

    recommendation = models.TextField(blank=True)

    scanned_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-scanned_at']
        verbose_name = 'Offer Scan History'
        verbose_name_plural = 'Offer Scan Histories'

    def __str__(self):
        return f"OfferScan #{self.id} - {self.offer_category} ({self.risk_level})"
