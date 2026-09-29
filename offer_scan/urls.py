from django.urls import path
from . import views

urlpatterns = [
    path('offer-scan/', views.offer_scan, name='offer_scan'),
]