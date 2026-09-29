from django.db.models.signals import post_save, post_delete
from django.dispatch import receiver

from accounts.models import FinancialProfile
from banking.models import Transaction, FixedDeposit, RecurringDeposit, PassionFund
from .models import AIInsight, AIExplanation

def clear_user_cache(user):
    """Helper to delete all cached insights and explanations for a user."""
    if user:
        try:
            AIInsight.objects.filter(user=user).delete()
            AIExplanation.objects.filter(user=user).delete()
            print(f"Cleared AI insights cache for user: {user.customer_id}")
        except Exception as e:
            print(f"Error clearing cache: {e}")

@receiver([post_save, post_delete], sender=FinancialProfile)
def invalidate_profile_cache(sender, instance, **kwargs):
    clear_user_cache(instance.user)

@receiver([post_save, post_delete], sender=Transaction)
def invalidate_transaction_cache(sender, instance, **kwargs):
    clear_user_cache(instance.user)

@receiver([post_save, post_delete], sender=FixedDeposit)
def invalidate_fd_cache(sender, instance, **kwargs):
    clear_user_cache(instance.user)

@receiver([post_save, post_delete], sender=RecurringDeposit)
def invalidate_rd_cache(sender, instance, **kwargs):
    clear_user_cache(instance.user)

@receiver([post_save, post_delete], sender=PassionFund)
def invalidate_passion_fund_cache(sender, instance, **kwargs):
    clear_user_cache(instance.user)
