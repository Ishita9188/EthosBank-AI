from django.urls import path
from . import views

urlpatterns = [
    path(
        'scheduled-withdrawal/',
        views.scheduled_withdrawal_view,
        name='scheduled_withdrawal'
    ),
    path(
        'scheduled-withdrawal/cancel/<int:withdrawal_id>/',
        views.cancel_withdrawal,
        name='cancel_withdrawal'
    ),
    path(
        'withdrawal-history/',
        views.withdrawal_history_view,
        name='withdrawal_history'
    ),
]
