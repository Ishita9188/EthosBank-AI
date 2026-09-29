from datetime import date

from django.db import models
from accounts.models import UserAccount

class Wallet(models.Model):

    user = models.OneToOneField(
        UserAccount,
        on_delete=models.CASCADE
    )

    balance = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return self.user.customer_id

class Transaction(models.Model):

    TRANSACTION_TYPES = [

        ('Deposit','Deposit'),
        ('Withdraw','Withdraw'),
        ('Transfer','Transfer')

    ]
    CATEGORY_CHOICES = [

    ('Deposit','Deposit'),
    ('Withdrawal','Withdrawal'),
    ('Transfer','Transfer'),

    ('Food','Food'),
    ('Shopping','Shopping'),
    ('Travel','Travel'),
    ('Bills','Bills'),
    ('Healthcare','Healthcare'),
    ('Education','Education'),

    ('Investment','Investment'),
    ('Subscription','Subscription'),

    ('Goal Saving','Goal Saving')

    ]
    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE
    )

    transaction_type = models.CharField(
        max_length=20,
        choices=TRANSACTION_TYPES
    )

    amount = models.DecimalField(
        max_digits=15,
        decimal_places=2
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )
    category = models.CharField(
        max_length=50,
        default='Other'
    )
    merchant_name = models.CharField(
        max_length=200,
        blank=True
    )
    merchant_type = models.CharField(
    max_length=100,
    blank=True
)
    description = models.TextField(
        blank=True
    )

    recipient = models.CharField(
        max_length=100,
        blank=True
    )

    location = models.CharField(
        max_length=100,
        blank=True
    )
    status = models.CharField(
    max_length=20,
    default='Completed'
)
    risk_score = models.FloatField(
    default=0
)
    channel = models.CharField(
    max_length=50,
    default='Web'
)

    def __str__(self):
        return self.transaction_type
    risk_flag = models.BooleanField(
        default=False
    )

    is_recurring = models.BooleanField(
        default=False
    )

class FixedDeposit(models.Model):

    user = models.ForeignKey(
        'accounts.UserAccount',
        on_delete=models.CASCADE
    )

    amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    tenure_months = models.IntegerField()

    interest_rate = models.DecimalField(
        max_digits=5,
        decimal_places=2
    )

    maturity_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    maturity_date = models.DateField()

    status = models.CharField(
        max_length=20,
        default='Active'
    )

    def __str__(self):
        return f"{self.user.customer_id} - FD"

class RecurringDeposit(models.Model):

    user = models.ForeignKey(
        'accounts.UserAccount',
        on_delete=models.CASCADE
    )

    monthly_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )
    next_due_date = models.DateField(
    default=date.today
)
    tenure_months = models.IntegerField()
    
    interest_rate = models.DecimalField(
        max_digits=5,
        decimal_places=2
    )

    maturity_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    maturity_date = models.DateField()

    status = models.CharField(
        max_length=20,
        default='Active'
    )
    installments_paid = models.IntegerField(
    default=0
)

    total_installments = models.IntegerField(
    default=0
)

    def __str__(self):
        return f"RD - {self.user.customer_id}"
    
class PassionFund(models.Model):

    user = models.ForeignKey(
        'accounts.UserAccount',
        on_delete=models.CASCADE
    )

    fund_name = models.CharField(
        max_length=100
    )

    passion_type = models.CharField(
        max_length=100
    )

    monthly_installment = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    target_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2
    )

    current_amount = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0
    )

    topups_this_month = models.IntegerField(
        default=0
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )

    status = models.CharField(
        max_length=20,
        default='Active'
    )

    def __str__(self):
        return self.fund_name

class ScheduledWithdrawal(models.Model):
    STATUS_CHOICES = [
        ('Scheduled', 'Scheduled'),
        ('Completed', 'Completed'),
        ('Cancelled', 'Cancelled'),
    ]
    
    PURPOSE_CHOICES = [
        ('Personal', 'Personal'),
        ('Bill Payment', 'Bill Payment'),
        ('Investment', 'Investment'),
        ('Salary', 'Salary'),
        ('Other', 'Other'),
    ]
    
    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE
    )
    
    amount = models.DecimalField(
        max_digits=15,
        decimal_places=2
    )
    
    scheduled_datetime = models.DateTimeField()
    
    purpose = models.CharField(
        max_length=50,
        choices=PURPOSE_CHOICES
    )
    
    remarks = models.TextField(
        blank=True,
        null=True
    )
    
    status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default='Scheduled'
    )
    
    created_at = models.DateTimeField(
        auto_now_add=True
    )

    def __str__(self):
        return f"{self.user.customer_id} - ₹{self.amount} - {self.status}"


class Beneficiary(models.Model):
    """Saved transfer recipients for a customer."""

    user = models.ForeignKey(
        UserAccount,
        on_delete=models.CASCADE,
        related_name='beneficiaries'
    )

    name = models.CharField(max_length=100)

    account_number = models.CharField(max_length=30)

    ifsc = models.CharField(max_length=11)

    bank_name = models.CharField(max_length=100)

    nickname = models.CharField(max_length=50, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'account_number')
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.name} ({self.account_number}) — {self.user.customer_id}"