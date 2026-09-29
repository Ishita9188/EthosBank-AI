from django.urls import path
from . import views

urlpatterns = [
    path('financial-wellness/', views.wellness_dashboard, name='financial_wellness'),
]
