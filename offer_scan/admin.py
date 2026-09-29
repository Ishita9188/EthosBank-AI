from django.contrib import admin
from .models import OfferScanHistory


@admin.register(OfferScanHistory)
class OfferScanHistoryAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user_id',
        'offer_category',
        'risk_level',
        'risk_score',
        'ethical_classification',
        'scanned_at'
    )
    list_filter = (
        'risk_level',
        'ethical_classification',
        'offer_category',
        'scanned_at'
    )
    search_fields = (
        'original_text',
        'recommendation'
    )
