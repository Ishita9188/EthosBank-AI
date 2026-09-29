from django.db import models

class UserAccount(models.Model):

    ROLE_CHOICES = [
        ('consumer', 'Consumer'),
        ('elder', 'Elder User')
    ]

    customer_id = models.CharField(max_length=20, unique=True)
    account_number = models.CharField(max_length=20, unique=True)

    role = models.CharField(max_length=20, choices=ROLE_CHOICES)

    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)

    email = models.EmailField(unique=True)
    mobile = models.CharField(max_length=15)

    dob = models.DateField()

    gender = models.CharField(max_length=20)

    occupation = models.CharField(max_length=100, blank=True)
    monthly_income = models.DecimalField(
        max_digits=12,
        decimal_places=2,
        default=0
    )

    password = models.CharField(max_length=255)

    trusted_contact_name = models.CharField(
        max_length=100,
        blank=True
    )

    trusted_contact_phone = models.CharField(
        max_length=15,
        blank=True
    )

    relationship = models.CharField(
        max_length=50,
        blank=True
    )
    aadhaar_document = models.ImageField(
        upload_to="identity_documents/aadhaar/",
        blank=True,
        null=True
    )

    pan_document = models.ImageField(
        upload_to="identity_documents/pan/",
        blank=True,
        null=True
    )

    aadhaar_verified = models.BooleanField(
        default=False
    )

    pan_verified = models.BooleanField(
        default=False
)

    aadhaar_last_four = models.CharField(
        max_length=4,
        blank=True
    )

    pan_number = models.CharField(
        max_length=10,
        blank=True
    )

    aadhaar_ocr_confidence = models.FloatField(
    default=0
)

    pan_ocr_confidence = models.FloatField(
    default=0
)

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.customer_id

class FinancialProfile(models.Model):

    EMPLOYMENT_CHOICES = [
        ('Student', 'Student'),
        ('Salaried', 'Salaried'),
        ('Self-Employed', 'Self-Employed'),
        ('Business Owner', 'Business Owner'),
        ('Freelancer', 'Freelancer'),
        ('Other', 'Other'),
    ]

    MARITAL_STATUS_CHOICES = [
        ('Single', 'Single'),
        ('Married', 'Married'),
        ('Divorced', 'Divorced'),
        ('Widowed', 'Widowed'),
    ]

    user = models.OneToOneField(
        UserAccount,
        on_delete=models.CASCADE
    )

    # Section 2: Employment & Income
    employment_type = models.CharField(
        max_length=50,
        choices=EMPLOYMENT_CHOICES,
        default='Salaried'
    )
    occupation = models.CharField(
        max_length=100
    )
    monthly_income = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )
    other_monthly_income = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0,
        blank=True,
        null=True
    )

    # Section 3: Family Details
    marital_status = models.CharField(
        max_length=30,
        choices=MARITAL_STATUS_CHOICES,
        default='Single'
    )
    dependents = models.PositiveIntegerField(
        default=0
    )

    # Section 4: Financial Goals
    # Storing as comma separated values or JSON string for simplicity in this case
    financial_goals = models.TextField(
        blank=True,
        help_text="Comma separated goals"
    )
    other_financial_goal = models.CharField(
        max_length=255,
        blank=True,
        null=True
    )

    # Section 5: Savings Habits
    monthly_savings_target = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )
    existing_savings = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )

    # Section 6: Liabilities
    home_loan = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )
    education_loan = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )
    personal_loan = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )
    credit_card_debt = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )
    other_debt = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )

    # Section 7: AI Personalization
    preferred_monthly_budget = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )
    preferred_savings_percentage = models.DecimalField(
        max_digits=5,
        decimal_places=2,
        default=0
    )
    spending_alert_limit = models.DecimalField(
        max_digits=15,
        decimal_places=2,
        default=0
    )
    receive_ai_recommendations = models.BooleanField(
        default=True
    )

    created_at = models.DateTimeField(
        auto_now_add=True
    )
    updated_at = models.DateTimeField(
        auto_now=True
    )

    def __str__(self):
        return f"{self.user.customer_id} Profile"