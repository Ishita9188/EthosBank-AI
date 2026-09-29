from django.db import transaction
from banking.models import Wallet
from django.db import IntegrityError
from analytics_engine.models import Beneficiary
from django.shortcuts import get_object_or_404
from django.http import request
from analytics_engine.models import AIInsight
from django.shortcuts import render, redirect
from decimal import Decimal
from .models import UserAccount, FinancialProfile
from banking.models import Wallet
from banking.models import Transaction, PassionFund, ScheduledWithdrawal
from django.db.models import Sum, Max, Avg
from django.core.paginator import Paginator
from random import randint
from django.contrib import messages
from banking.models import FixedDeposit
from datetime import date, timedelta, datetime
from dateutil.relativedelta import relativedelta
from banking.models import RecurringDeposit
from django.utils import timezone
from django.views.decorators.csrf import ensure_csrf_cookie
import hashlib
import requests
from django.conf import settings
from .document_analyzer import (
    analyze_aadhaar,
    analyze_pan
)

def home(request):
    return render(request, "landing.html")
@ensure_csrf_cookie
def register_view(request):

    # =========================================================
    # GET REQUEST
    # =========================================================

    if request.method == "GET":

        return render(
            request,
            "register.html",
            {
                "recaptcha_site_key": settings.RECAPTCHA_SITE_KEY
            }
        )

    # =========================================================
    # POST REQUEST
    # =========================================================

    if request.method == "POST":

        # -----------------------------------------------------
        # BASIC FORM DATA
        # -----------------------------------------------------

        role = request.POST.get("role", "").strip()

        first_name = request.POST.get(
            "first_name",
            ""
        ).strip()

        last_name = request.POST.get(
            "last_name",
            ""
        ).strip()

        email = request.POST.get(
            "email",
            ""
        ).strip().lower()

        mobile = request.POST.get(
            "mobile",
            ""
        ).strip()

        dob = request.POST.get(
            "dob",
            ""
        )

        gender = request.POST.get(
            "gender",
            ""
        )

        occupation = request.POST.get(
            "occupation",
            ""
        ).strip()

        monthly_income = request.POST.get(
            "monthly_income",
            "0"
        )

        password = request.POST.get(
            "password",
            ""
        )

        confirm_password = request.POST.get(
            "confirm_password",
            ""
        )

        # -----------------------------------------------------
        # ROLE VALIDATION
        # -----------------------------------------------------

        if role not in [
            "consumer",
            "elder"
        ]:

            messages.error(
                request,
                "Please select a valid user type."
            )

            return redirect("register")

        # -----------------------------------------------------
        # PASSWORD VALIDATION
        # -----------------------------------------------------

        if password != confirm_password:

            messages.error(
                request,
                "Passwords do not match."
            )

            return redirect("register")

        if len(password) < 8:

            messages.error(
                request,
                "Password must contain at least 8 characters."
            )

            return redirect("register")

        # -----------------------------------------------------
        # GOOGLE reCAPTCHA
        # -----------------------------------------------------

        captcha_response = request.POST.get(
            "g-recaptcha-response"
        )

        if not captcha_response:

            messages.error(
                request,
                "Please complete the Google reCAPTCHA."
            )

            return redirect("register")

        captcha_data = {
            "secret": settings.RECAPTCHA_SECRET_KEY,
            "response": captcha_response,
            "remoteip": request.META.get(
                "REMOTE_ADDR"
            )
        }

        try:

            captcha_response_data = requests.post(
                "https://www.google.com/recaptcha/api/siteverify",
                data=captcha_data,
                timeout=10
            )

            captcha_result = captcha_response_data.json()

        except requests.RequestException:

            messages.error(
                request,
                "Unable to verify CAPTCHA. Please try again."
            )

            return redirect("register")

        if not captcha_result.get(
            "success",
            False
        ):

            messages.error(
                request,
                "Google reCAPTCHA verification failed."
            )

            return redirect("register")

        # -----------------------------------------------------
        # TERMS AND CONDITIONS
        # -----------------------------------------------------

        terms_accepted = request.POST.get(
            "terms"
        )

        if terms_accepted != "on":

            messages.error(
                request,
                "You must accept the Terms & Conditions."
            )

            return redirect("register")

        # -----------------------------------------------------
        # EMAIL DUPLICATE CHECK
        # -----------------------------------------------------

        if UserAccount.objects.filter(
            email=email
        ).exists():

            messages.error(
                request,
                "An account with this email already exists."
            )

            return redirect("register")

        # -----------------------------------------------------
        # IDENTITY DOCUMENTS
        # -----------------------------------------------------

        aadhaar_file = request.FILES.get(
            "aadhaar_document"
        )

        pan_file = request.FILES.get(
            "pan_document"
        )

        if not aadhaar_file:

            messages.error(
                request,
                "Please upload your Aadhaar document."
            )

            return redirect("register")

        if not pan_file:

            messages.error(
                request,
                "Please upload your PAN document."
            )

            return redirect("register")

        # -----------------------------------------------------
        # FILE TYPE VALIDATION
        # -----------------------------------------------------

        allowed_types = [
            "image/jpeg",
            "image/png",
            "image/jpg"
        ]

        if aadhaar_file.content_type not in allowed_types:

            messages.error(
                request,
                "Aadhaar must be a JPG or PNG image."
            )

            return redirect("register")

        if pan_file.content_type not in allowed_types:

            messages.error(
                request,
                "PAN must be a JPG or PNG image."
            )

            return redirect("register")

        # -----------------------------------------------------
        # FILE SIZE VALIDATION
        # -----------------------------------------------------

        max_size = 5 * 1024 * 1024

        if aadhaar_file.size > max_size:

            messages.error(
                request,
                "Aadhaar image must be below 5 MB."
            )

            return redirect("register")

        if pan_file.size > max_size:

            messages.error(
                request,
                "PAN image must be below 5 MB."
            )

            return redirect("register")

        # -----------------------------------------------------
        # TEMPORARILY SAVE FILES FOR OCR
        # -----------------------------------------------------

        from django.core.files.storage import default_storage

        aadhaar_path = None
        pan_path = None

        try:

            aadhaar_path = default_storage.save(
                "identity_documents/temp/" + aadhaar_file.name,
                aadhaar_file
            )

            pan_path = default_storage.save(
                "identity_documents/temp/" + pan_file.name,
                pan_file
            )

            aadhaar_full_path = default_storage.path(
                aadhaar_path
            )

            pan_full_path = default_storage.path(
                pan_path
            )

            # -------------------------------------------------
            # OCR ANALYSIS
            # -------------------------------------------------

            try:

                aadhaar_result = analyze_aadhaar(
                    aadhaar_full_path
                )

                pan_result = analyze_pan(
                    pan_full_path
                )

            except Exception as e:

                print(
                    "DOCUMENT OCR ERROR:",
                    e
                )

                messages.error(
                    request,
                    "Unable to analyse identity documents."
                )

                return redirect("register")

            # -------------------------------------------------
            # DOCUMENT VALIDATION
            # -------------------------------------------------

            if not aadhaar_result.get(
                "document_detected",
                False
            ):

                messages.error(
                    request,
                    "The uploaded Aadhaar document could not be detected."
                )

                return redirect("register")

            if not pan_result.get(
                "document_detected",
                False
            ):

                messages.error(
                    request,
                    "The uploaded PAN document could not be detected."
                )

                return redirect("register")

            # -------------------------------------------------
            # PASSWORD HASH
            # -------------------------------------------------

            hashed_password = hashlib.sha256(
                password.encode()
            ).hexdigest()

            # =================================================
            # UNIQUE CUSTOMER ID
            # =================================================

            # Do NOT rely only on count().
            #
            # Example:
            #
            # ETH202600001
            # ETH202600002
            #
            # If a record is deleted, count() can produce
            # an ID that already existed.

            user_count = UserAccount.objects.count() + 1

            while True:

                customer_id = (
                    f"ETH2026{user_count:05d}"
                )

                if not UserAccount.objects.filter(
                    customer_id=customer_id
                ).exists():

                    break

                user_count += 1

            # =================================================
            # UNIQUE ACCOUNT NUMBER
            # =================================================

            #
            # Your old code:
            #
            # count = UserAccount.objects.count() + 1
            # account_number = f"7845{10000000 + count}"
            #
            # is NOT guaranteed to be unique.
            #
            # We keep your same format but explicitly check
            # the database before using it.
            #

            account_counter = (
                UserAccount.objects.count() + 1
            )

            while True:

                account_number = (
                    f"7845{10000000 + account_counter}"
                )

                if not UserAccount.objects.filter(
                    account_number=account_number
                ).exists():

                    break

                account_counter += 1

            # -------------------------------------------------
            # CREATE USER
            # -------------------------------------------------

            try:

                with transaction.atomic():

                    user = UserAccount(

                        customer_id=customer_id,

                        account_number=account_number,

                        first_name=first_name,

                        last_name=last_name,

                        email=email,

                        mobile=mobile,

                        dob=dob,

                        gender=gender,

                        occupation=occupation,

                        monthly_income=monthly_income,

                        role=role,

                        password=hashed_password,

                        aadhaar_document=aadhaar_file,

                        pan_document=pan_file,

                        aadhaar_verified=True,

                        pan_verified=True,

                        aadhaar_last_four=
                            aadhaar_result.get(
                                "aadhaar_last_four",
                                ""
                            ),

                        pan_number=
                            pan_result.get(
                                "pan_number",
                                ""
                            ),

                        aadhaar_ocr_confidence=
                            aadhaar_result.get(
                                "confidence",
                                0
                            ),

                        pan_ocr_confidence=
                            pan_result.get(
                                "confidence",
                                0
                            )
                    )

                    # -----------------------------------------
                    # ELDER USER DETAILS
                    # -----------------------------------------

                    if role == "elder":

                        user.trusted_contact_name = (
                            request.POST.get(
                                "trusted_contact_name",
                                ""
                            ).strip()
                        )

                        user.trusted_contact_phone = (
                            request.POST.get(
                                "trusted_contact_phone",
                                ""
                            ).strip()
                        )

                        user.relationship = (
                            request.POST.get(
                                "relationship",
                                ""
                            ).strip()
                        )

                    user.save()
                    Wallet.objects.get_or_create(
                        user=user,
                        defaults={
                            'balance': 0
                        }
                    )

            except IntegrityError as e:

                print(
                    "REGISTRATION DATABASE ERROR:",
                    e
                )

                messages.error(
                    request,
                    "Unable to create the account because a unique "
                    "customer or account number conflict occurred. "
                    "Please try registering again."
                )

                return redirect("register")

            # -------------------------------------------------
            # REMOVE TEMPORARY OCR FILES
            # -------------------------------------------------

            try:

                if aadhaar_path:
                    default_storage.delete(
                        aadhaar_path
                    )

                if pan_path:
                    default_storage.delete(
                        pan_path
                    )

            except Exception:

                pass

            # -------------------------------------------------
            # SUCCESS
            # -------------------------------------------------

            messages.success(
                request,
                f"Registration successful! "
                f"Your Customer ID is {customer_id}. "
                f"Your Account Number is {account_number}."
            )

            return redirect("login")

        except Exception as e:

            print(
                "REGISTRATION ERROR:",
                e
            )

            messages.error(
                request,
                "Registration failed. Please try again."
            )

            return redirect("register")

    # =========================================================
    # FALLBACK
    # =========================================================

    return redirect("register")
def register(request):

    if 'captcha_a' not in request.session:

        request.session['captcha_a'] = randint(1, 10)
        request.session['captcha_b'] = randint(1, 10)

    a = request.session['captcha_a']
    b = request.session['captcha_b']

    if request.method == 'POST':

        captcha = int(
            request.POST.get('captcha_answer')
        )

        if captcha != (a + b):

            messages.error(
                request,
                "Invalid CAPTCHA"
            )

            return redirect('register')

        count = UserAccount.objects.count() + 1

        customer_id = f"ETH2026{count:05d}"

        account_number = f"7845{10000000 + count}"

        while UserAccount.objects.filter(
            account_number=account_number
        ).exists():

            count += 1

            customer_id = f"ETH2026{count:05d}"

            account_number = f"7845{10000000 + count}"

        password = request.POST['password']

        hashed_password = hashlib.sha256(
            password.encode()
        ).hexdigest()

        user = UserAccount.objects.create(

            customer_id=customer_id,
            account_number=account_number,

            role=request.POST['role'],

            first_name=request.POST['first_name'],
            last_name=request.POST['last_name'],

            email=request.POST['email'],
            mobile=request.POST['mobile'],

            dob=request.POST['dob'],

            gender=request.POST['gender'],

            occupation=request.POST['occupation'],

            monthly_income=request.POST.get(
                'monthly_income',
                0
            ),

            password=hashed_password,

            trusted_contact_name=request.POST.get(
                'trusted_contact_name',
                ''
            ),

            trusted_contact_phone=request.POST.get(
                'trusted_contact_phone',
                ''
            ),

            relationship=request.POST.get(
                'relationship',
                ''
            )

        )

        Wallet.objects.create(
            user=user,
            balance=0
        )

        request.session['customer_id'] = customer_id
        request.session['account_number'] = account_number

        return redirect('registration_success')

    context = {
        'captcha_question': f"{a} + {b}"
    }

    return render(
        request,
        'register.html',
        context
    )

def registration_success(request):

    customer_id = request.session.get('customer_id')
    account_number = request.session.get('account_number')

    context = {
        'customer_id': customer_id,
        'account_number': account_number
    }

    return render(
        request,
        'registration_success.html',
        context
    )
def login_view(request):

    # =========================================================
    # POST REQUEST - LOGIN
    # =========================================================

    if request.method == "POST":

        # -----------------------------------------------------
        # GET FORM VALUES
        # -----------------------------------------------------

        customer_id = request.POST.get(
            'customer_id',
            ''
        ).strip()

        password = request.POST.get(
            'password',
            ''
        )

        remember_me = request.POST.get(
            'remember_me'
        ) == 'on'


        # -----------------------------------------------------
        # GOOGLE reCAPTCHA
        # -----------------------------------------------------

        recaptcha_response = request.POST.get(
            'g-recaptcha-response'
        )

        if not recaptcha_response:

            messages.error(
                request,
                "Please complete the Google reCAPTCHA verification."
            )

            return render(
                request,
                'login.html',
                {
                    'recaptcha_site_key':
                        settings.RECAPTCHA_SITE_KEY
                }
            )


        # -----------------------------------------------------
        # VERIFY reCAPTCHA WITH GOOGLE
        # -----------------------------------------------------

        try:

            recaptcha_result = requests.post(
                'https://www.google.com/recaptcha/api/siteverify',

                data={
                    'secret':
                        settings.RECAPTCHA_SECRET_KEY,

                    'response':
                        recaptcha_response,

                    'remoteip':
                        request.META.get(
                            'REMOTE_ADDR'
                        )
                },

                timeout=10
            )

            recaptcha_data = recaptcha_result.json()


        except requests.RequestException:

            messages.error(
                request,
                "Unable to connect to Google reCAPTCHA. "
                "Please try again."
            )

            return render(
                request,
                'login.html',
                {
                    'recaptcha_site_key':
                        settings.RECAPTCHA_SITE_KEY
                }
            )


        # -----------------------------------------------------
        # CHECK GOOGLE RESPONSE
        # -----------------------------------------------------

        if not recaptcha_data.get('success', False):

            messages.error(
                request,
                "Invalid reCAPTCHA verification. Please try again."
            )

            return render(
                request,
                'login.html',
                {
                    'recaptcha_site_key':
                        settings.RECAPTCHA_SITE_KEY
                }
            )


        # -----------------------------------------------------
        # VALIDATE CUSTOMER ID
        # -----------------------------------------------------

        if not customer_id:

            messages.error(
                request,
                "Please enter your Customer ID."
            )

            return render(
                request,
                'login.html',
                {
                    'recaptcha_site_key':
                        settings.RECAPTCHA_SITE_KEY
                }
            )


        # -----------------------------------------------------
        # VALIDATE PASSWORD
        # -----------------------------------------------------

        if not password:

            messages.error(
                request,
                "Please enter your password."
            )

            return render(
                request,
                'login.html',
                {
                    'recaptcha_site_key':
                        settings.RECAPTCHA_SITE_KEY
                }
            )


        # -----------------------------------------------------
        # SHA-256 PASSWORD HASH
        # -----------------------------------------------------
        #
        # KEEPING YOUR EXISTING PASSWORD SYSTEM
        #

        password_hash = hashlib.sha256(
            password.encode()
        ).hexdigest()


        # -----------------------------------------------------
        # CHECK USER CREDENTIALS
        # -----------------------------------------------------

        try:

            user = UserAccount.objects.get(
                customer_id=customer_id,
                password=password_hash
            )


        except UserAccount.DoesNotExist:

            messages.error(
                request,
                "Invalid Credentials"
            )

            return render(
                request,
                'login.html',
                {
                    'recaptcha_site_key':
                        settings.RECAPTCHA_SITE_KEY
                }
            )


        # -----------------------------------------------------
        # LOGIN SUCCESS
        # -----------------------------------------------------

        request.session['user_id'] = user.id

        request.session['role'] = user.role


        # -----------------------------------------------------
        # REMEMBER ME
        # -----------------------------------------------------
        #
        # CHECKED:
        # Session remains for 30 days.
        #
        # UNCHECKED:
        # Session expires when browser closes.
        #

        if remember_me:

            request.session.set_expiry(
                60 * 60 * 24 * 30
            )

        else:

            request.session.set_expiry(0)


        # -----------------------------------------------------
        # REDIRECT BASED ON ROLE
        # -----------------------------------------------------

        if user.role == 'consumer':

            return redirect(
                'consumer_dashboard'
            )


        elif user.role == 'elder':

            return redirect(
                'elder_dashboard'
            )


        else:

            messages.error(
                request,
                "Invalid user role."
            )

            request.session.flush()

            return redirect(
                'login'
            )


    # =========================================================
    # GET REQUEST - SHOW LOGIN PAGE
    # =========================================================

    return render(
        request,
        'login.html',
        {
            'recaptcha_site_key':
                settings.RECAPTCHA_SITE_KEY
        }
    )

def consumer_dashboard(request):

    user_id = request.session.get('user_id')

    user = UserAccount.objects.get(
        id=user_id
    )


    wallet, created = Wallet.objects.get_or_create(
    user=user,
    defaults={
        "balance": 0
    }
)

    transactions = Transaction.objects.filter(
    user=user
).order_by('-created_at')[:5]

    due_withdrawals = ScheduledWithdrawal.objects.filter(
        user=user,
        scheduled_datetime__lte=timezone.now(),
        status='Scheduled'
    ).order_by('scheduled_datetime')
    
    due_withdrawals_count = due_withdrawals.count()
    single_due_withdrawal = None
    formatted_due_time = ""
    
    if due_withdrawals_count == 1:
        single_due_withdrawal = due_withdrawals.first()
        local_dt = timezone.localtime(single_due_withdrawal.scheduled_datetime)
        now = timezone.localtime(timezone.now())
        time_str = local_dt.strftime("%I:%M %p")
        if local_dt.date() == now.date():
            formatted_due_time = f"today at {time_str}"
        elif local_dt.date() == now.date() - timedelta(days=1):
            formatted_due_time = f"yesterday at {time_str}"
        else:
            formatted_due_time = f"on {local_dt.strftime('%d-%b-%Y')} at {time_str}"

    from analytics_engine.models import AIInsight, AIExplanation

    wellness = AIInsight.objects.filter(
        user=user,
        insight_type="Financial Wellness"
    ).first()

    debt = AIInsight.objects.filter(
        user=user,
        insight_type="Debt Risk"
    ).first()

    debt_exp = AIExplanation.objects.filter(
        user=user,
        model_name="Debt Risk"
    ).first()

    context = {
        "wallet": wallet,
        "transactions": transactions,
        "wellness": wellness,
        "debt": debt,
        "debt_exp": debt_exp,
    }

    context = {
        
        'user': user,
        'wallet': wallet,
        'transactions': transactions,
        'due_withdrawals_count': due_withdrawals_count,
        'single_due_withdrawal': single_due_withdrawal,
        'formatted_due_time': formatted_due_time,
        'wellness': wellness,
        'debt': debt,
        'debt_exp': debt_exp
    }

    return render(
        request,
        'consumer_dashboard.html',
        context
    )

def wallet_page(request):

    user_id = request.session.get('user_id')

    user = UserAccount.objects.get(
        id=user_id
    )

    wallet, created = Wallet.objects.get_or_create(
    user=user,
    defaults={
        "balance": 0
    }
)

    transactions = Transaction.objects.filter(
        user=user
    ).order_by('-created_at')[:10]

    total_deposit = Transaction.objects.filter(
        user=user,
        transaction_type='Deposit'
    ).aggregate(
        Sum('amount')
    )['amount__sum'] or 0

    total_withdraw = Transaction.objects.filter(
        user=user,
        transaction_type='Withdraw'
    ).aggregate(
        Sum('amount')
    )['amount__sum'] or 0

    total_transfer = Transaction.objects.filter(
        user=user,
        transaction_type='Transfer'
    ).aggregate(
        Sum('amount')
    )['amount__sum'] or 0

    context = {

        'user': user,
        'wallet': wallet,
        'transactions': transactions,

        'total_deposit': total_deposit,
        'total_withdraw': total_withdraw,
        'total_transfer': total_transfer

    }

    return render(
        request,
        'wallet.html',
        context
    )
def deposit_page(request):

    user_id = request.session.get('user_id')

    user = UserAccount.objects.get(
        id=user_id
    )

    wallet = Wallet.objects.get(
        user=user
    )

    if request.method == "POST":

        amount = Decimal(
            request.POST.get('amount')
        )

        if amount <= 0:

            messages.error(
                request,
                "Invalid Amount"
            )

            return redirect('deposit')

        wallet.balance += amount

        wallet.save()

        Transaction.objects.create(

    user=user,

    transaction_type='Deposit',

    amount=amount,

    category='Deposit',

    merchant_name='Self Deposit',

    merchant_type='Banking',

    description='Wallet Deposit',

    channel='Web',

    status='Completed',

    risk_score=0,

    risk_flag=False,

    is_recurring=False

)

        messages.success(
            request,
            f'₹{amount} deposited successfully'
        )

        return redirect('wallet')

    return render(
        request,
        'deposit.html',
        {
            'wallet': wallet
        }
    )

def fixed_deposit(request):
    return render(request, 'fixed_deposit.html')   
def recurring_deposit(request):
    return render(request, 'recurring_deposit.html')
def passion_fund(request):
    return render(request, 'passion_fund.html')
def all_deposits(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')

    user = UserAccount.objects.get(id=user_id)

    # Base queryset — all deposit transactions for this user
    all_dep_qs = Transaction.objects.filter(
        user=user,
        transaction_type='Deposit'
    )

    # ── Top Statistics (always computed on full history) ──────────────────────
    total_deposits = all_dep_qs.aggregate(Sum('amount'))['amount__sum'] or Decimal('0.00')
    largest_deposit = all_dep_qs.aggregate(Max('amount'))['amount__max'] or Decimal('0.00')
    avg_deposit = all_dep_qs.aggregate(Avg('amount'))['amount__avg'] or Decimal('0.00')

    now = timezone.now()
    this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    this_month_deposits = (
        all_dep_qs
        .filter(created_at__gte=this_month_start)
        .aggregate(Sum('amount'))['amount__sum'] or Decimal('0.00')
    )

    # ── Filters ───────────────────────────────────────────────────────────────
    filtered_qs = all_dep_qs

    start_date_str = request.GET.get('start_date', '')
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            filtered_qs = filtered_qs.filter(created_at__date__gte=start_date)
        except ValueError:
            pass

    end_date_str = request.GET.get('end_date', '')
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            filtered_qs = filtered_qs.filter(created_at__date__lte=end_date)
        except ValueError:
            pass

    min_amount_str = request.GET.get('min_amount', '')
    if min_amount_str:
        try:
            filtered_qs = filtered_qs.filter(amount__gte=Decimal(min_amount_str))
        except Exception:
            pass

    max_amount_str = request.GET.get('max_amount', '')
    if max_amount_str:
        try:
            filtered_qs = filtered_qs.filter(amount__lte=Decimal(max_amount_str))
        except Exception:
            pass

    category_filter = request.GET.get('category', '')
    if category_filter:
        filtered_qs = filtered_qs.filter(category__icontains=category_filter)

    channel_filter = request.GET.get('channel', '')
    if channel_filter and channel_filter != 'All':
        filtered_qs = filtered_qs.filter(channel__iexact=channel_filter)

    # Build list of dicts for the template (keeps template format consistent)
    deposits_list = []
    for t in filtered_qs.order_by('-created_at'):
        deposits_list.append({
            'id': f"DEP-{t.id:06d}",
            'raw_id': t.id,
            'date': t.created_at,
            'amount': t.amount,
            'category': t.category,
            'channel': t.channel,
            'description': t.description or '',
            'status': t.status,
        })

    # ── Pagination ────────────────────────────────────────────────────────────
    paginator = Paginator(deposits_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    context = {
        'user': user,
        'page_obj': page_obj,
        'total_deposits': total_deposits,
        'this_month_deposits': this_month_deposits,
        'avg_deposit': avg_deposit,
        'largest_deposit': largest_deposit,
        'start_date': start_date_str,
        'end_date': end_date_str,
        'min_amount': min_amount_str,
        'max_amount': max_amount_str,
        'category': category_filter,
        'channel': channel_filter,
    }

    return render(request, 'all_deposits.html', context)
def withdraw_page(request):

    user_id = request.session.get('user_id')

    user = UserAccount.objects.get(
        id=user_id
    )

    wallet = Wallet.objects.get(
        user=user
    )

    if request.method == "POST":

        category = request.POST.get('category', 'Other')

        amount = Decimal(
            request.POST.get('amount')
        )

        if amount > wallet.balance:

            messages.error(
                request,
                "Insufficient Wallet Balance"
            )

            return redirect('withdraw')

        wallet.balance -= amount

        wallet.save()

        Transaction.objects.create(

    user=user,

    transaction_type='Withdraw',

    amount=amount,

    category=category,

    merchant_name='Self Withdrawal',

    merchant_type='Banking',

    description='Wallet Withdrawal',

    channel='Web',

    status='Completed',

    risk_score=5,

    risk_flag=False,

    is_recurring=False

)

        messages.success(
            request,
            f'₹{amount} withdrawn successfully'
        )

        return redirect('wallet')

    return render(
        request,
        'withdraw.html',
        {
            'wallet': wallet
        }
    )

def transfer_page(request):

    user_id = request.session.get('user_id')

    sender = UserAccount.objects.get(
        id=user_id
    )

    sender_wallet = Wallet.objects.get(
        user=sender
    )

    if request.method == "POST":

        category = request.POST.get('category', 'Other')

        customer_id = request.POST.get(
            'customer_id'
        )

        amount = Decimal(
            request.POST.get(
                'amount'
            )
        )

        try:

            receiver = UserAccount.objects.get(
                customer_id=customer_id
            )

        except UserAccount.DoesNotExist:

            messages.error(
                request,
                "Customer ID not found"
            )

            return redirect(
                'transfer'
            )

        if amount > sender_wallet.balance:

            messages.error(
                request,
                "Insufficient Balance"
            )

            return redirect(
                'transfer'
            )

        receiver_wallet = Wallet.objects.get(
            user=receiver
        )

        sender_wallet.balance -= amount
        receiver_wallet.balance += amount

        sender_wallet.save()
        receiver_wallet.save()

        Transaction.objects.create(

    user=sender,

    transaction_type='Transfer',

    amount=amount,

    category=category,

    recipient=receiver.customer_id,

    merchant_name=receiver.first_name,

    merchant_type='P2P',

    description='Fund Transfer',

    channel='Web',

    status='Completed',

    risk_score=10,

    risk_flag=False,

    is_recurring=False

)

        messages.success(
            request,
            "Transfer Successful"
        )

        return redirect(
            'wallet'
        )

    return render(
        request,
        'transfer.html',
        {
            'wallet': sender_wallet
        }
    )
def transactions_page(request):

    user_id = request.session.get(
        'user_id'
    )

    user = UserAccount.objects.get(
        id=user_id
    )

    transactions = Transaction.objects.filter(
        user=user
    ).order_by(
        '-created_at'
    )

    transaction_type = request.GET.get(
        'type'
    )

    if transaction_type:

        transactions = transactions.filter(
            transaction_type=transaction_type
        )

    total_deposit = Transaction.objects.filter(
        user=user,
        transaction_type='Deposit'
    ).aggregate(
        Sum('amount')
    )['amount__sum'] or 0

    total_withdraw = Transaction.objects.filter(
        user=user,
        transaction_type='Withdraw'
    ).aggregate(
        Sum('amount')
    )['amount__sum'] or 0

    total_transfer = Transaction.objects.filter(
        user=user,
        transaction_type='Transfer'
    ).aggregate(
        Sum('amount')
    )['amount__sum'] or 0

    return render(
        request,
        'transactions.html',
        {

            'transactions': transactions,

            'total_deposit': total_deposit,

            'total_withdraw': total_withdraw,

            'total_transfer': total_transfer,

            'total_transactions':
                transactions.count()

        }
    )
def fixed_deposit(request):

    user_id = request.session.get('user_id')

    user = UserAccount.objects.get(
        id=user_id
    )

    wallet = Wallet.objects.get(
        user=user
    )

    if request.method == "POST":

        amount = Decimal(
            request.POST.get('amount')
        )

        tenure = int(
            request.POST.get('tenure')
        )

        if amount > wallet.balance:

            messages.error(
                request,
                "Insufficient Balance"
            )

            return redirect(
                'fixed_deposit'
            )

        interest_rates = {

            6: 5.5,
            12: 6.5,
            24: 7.0,
            36: 7.5,
            60: 8.0

        }

        rate = interest_rates.get(
            tenure,
            6.5
        )

        maturity_amount = round(

            amount +

            (
                amount *   Decimal(str(rate)) * tenure ) / (100 * 12), 2

        )

        wallet.balance -= amount

        wallet.save()

        FixedDeposit.objects.create(

            user=user,

            amount=amount,

            tenure_months=tenure,

            interest_rate=rate,

            maturity_amount=maturity_amount,

            maturity_date=
            date.today() +
            relativedelta(
                months=tenure
            )

        )

        Transaction.objects.create(

    user=user,

    transaction_type='Fixed Deposit',

    amount=amount,

    category='Investment',

    merchant_name='Fixed Deposit',

    merchant_type='Investment',

    description=f'FD created for {tenure} months',

    channel='Web',

    status='Completed',

    risk_score=0,

    risk_flag=False,

    is_recurring=False

)

        messages.success(

            request,

            "Fixed Deposit Created Successfully"

        )

        return redirect(
            'fixed_deposit'
        )

    fds = FixedDeposit.objects.filter(
        user=user
    )

    return render(

        request,

        'fixed_deposit.html',

        {

            'wallet': wallet,

            'fds': fds

        }

    )

def recurring_deposit(request):

    user_id = request.session.get(
        'user_id'
    )

    user = UserAccount.objects.get(
        id=user_id
    )

    wallet = Wallet.objects.get(
        user=user
    )

    if request.method == "POST":

        monthly_amount = Decimal(
            request.POST.get(
                'monthly_amount'
            )
        )

        tenure = int(
            request.POST.get(
                'tenure'
            )
        )

        rates = {

            12: Decimal('6.5'),
            24: Decimal('7.0'),
            36: Decimal('7.5'),
            60: Decimal('8.0')

        }

        rate = rates.get(
            tenure,
            Decimal('6.5')
        )

        total_investment = (
            monthly_amount *
            Decimal(str(tenure))
        )

        maturity_amount = (
            total_investment +
            (
                total_investment *
                rate *
                Decimal(str(tenure))
            ) / Decimal('2400')
        )

        RecurringDeposit.objects.create(

            user=user,
            next_due_date= date.today() +relativedelta(months=1),
            monthly_amount=monthly_amount,

            tenure_months=tenure,
            total_installments=tenure,

            installments_paid=0,

            interest_rate=rate,

            maturity_amount=
            maturity_amount,

            maturity_date=
            date.today() +
            relativedelta(
                months=tenure
            )

        )

        messages.success(

            request,

            "Recurring Deposit Created Successfully"

        )

        return redirect(
            'recurring_deposit'
        )

    rds = RecurringDeposit.objects.filter(
        user=user
    )

    return render(

        request,

        'recurring_deposit.html',

        {

            'wallet': wallet,

            'rds': rds

        }

    )

def pay_rd_installment(request, rd_id):

    rd = RecurringDeposit.objects.get(
        id=rd_id
    )

    wallet = Wallet.objects.get(
        user=rd.user
    )

    if rd.installments_paid >= rd.total_installments:

        messages.info(

            request,

            "RD already completed."

        )

        return redirect(
            'recurring_deposit'
        )

    amount = rd.monthly_amount

    if wallet.balance < amount:

        messages.error(

            request,

            "Insufficient Wallet Balance."

        )

        return redirect(
            'recurring_deposit'
        )

    wallet.balance -= amount

    wallet.save()

    rd.installments_paid += 1
    rd.next_due_date += relativedelta( months=1)
    rd.save()

    Transaction.objects.create(

    user=rd.user,

    transaction_type='RD Installment',

    amount=amount,

    category='Investment',

    merchant_name='Recurring Deposit',

    merchant_type='Investment',

    description='Monthly RD Installment',

    channel='Web',

    status='Completed',

    risk_score=0,

    risk_flag=False,

    is_recurring=True

)

    messages.success(

        request,

        "Installment Paid Successfully"

    )

    return redirect(
        'recurring_deposit'
    )

# Add this import at the top of the file, alongside your other Django imports:
from django.db.models import Sum


def my_passion_fund(request):

    user_id = request.session.get(
        'user_id'
    )

    user = UserAccount.objects.get(
        id=user_id
    )

    wallet = Wallet.objects.get(
        user=user
    )

    if request.method == "POST":

        fund_name = request.POST.get(
            'fund_name'
        )

        passion_type = request.POST.get(
            'passion_type'
        )

        installment = Decimal(
            request.POST.get(
                'monthly_installment'
            )
        )

        target_amount = Decimal(
            request.POST.get(
                'target_amount'
            )
        )

        if installment < 1000:

            messages.error(
                request,
                "Minimum installment is ₹1000"
            )

            return redirect(
                'my_passion_fund'
            )

        PassionFund.objects.create(

            user=user,

            fund_name=fund_name,

            passion_type=passion_type,

            monthly_installment=installment,

            target_amount=target_amount

        )

        messages.success(

            request,

            "Passion Fund Created Successfully"

        )

        return redirect(
            'my_passion_fund'
        )

    funds = PassionFund.objects.filter(
        user=user
    )

    totals = funds.aggregate(
        total_saved=Sum('current_amount'),
        total_target=Sum('target_amount')
    )

    total_saved = totals['total_saved'] or 0

    total_target = totals['total_target'] or 0

    if total_target > 0:

        progress_percent = round(
            (total_saved / total_target) * 100,
            1
        )

    else:

        progress_percent = 0

    return render(

        request,

        'my_passion_fund.html',

        {

            'wallet': wallet,

            'funds': funds,

            'total_saved': total_saved,

            'progress_percent': progress_percent

        }

    )

def passion_fund_topup(
        request,
        fund_id
):

    fund = PassionFund.objects.get(
        id=fund_id
    )

    wallet = Wallet.objects.get(
        user=fund.user
    )

    if request.method == "POST":

        amount = Decimal(
            request.POST.get(
                'amount'
            )
        )

        max_topup = (
            fund.monthly_installment * 2
        )

        if amount < 1000:

            messages.error(
                request,
                "Minimum top-up ₹1000"
            )

            return redirect(
                'my_passion_fund'
            )

        if amount > max_topup:

            messages.error(

                request,

                f"Maximum allowed ₹{max_topup}"

            )

            return redirect(
                'my_passion_fund'
            )

        if wallet.balance < amount:

            messages.error(

                request,

                "Insufficient Balance"

            )

            return redirect(
                'my_passion_fund'
            )

        wallet.balance -= amount

        wallet.save()

        fund.current_amount += amount

        fund.topups_this_month += 1

        fund.save()

        Transaction.objects.create(

    user=fund.user,

    transaction_type='Passion Fund Topup',

    amount=amount,

    category='Goal Saving',

    merchant_name=fund.fund_name,

    merchant_type='Savings Goal',

    description=f'Topup for {fund.fund_name}',

    channel='Web',

    status='Completed',

    risk_score=0,

    risk_flag=False,

    is_recurring=False

)

        messages.success(

            request,

            "Top-up Successful"

        )

    return redirect(
        'my_passion_fund'
    )

def financial_profile_view(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')
    user = UserAccount.objects.get(id=user_id)
    profile = FinancialProfile.objects.filter(user=user).first()
    receive_ai = True

    if profile and profile.receive_ai_recommendations is not None:
        receive_ai = profile.receive_ai_recommendations
    
    profile, created = FinancialProfile.objects.get_or_create(user=user)

# --------------------------------------------------
# Automatically calculate Existing Savings
# --------------------------------------------------

    wallet = Wallet.objects.filter(user=user).first()
    wallet_balance = float(wallet.balance) if wallet else 0

    fd_total = float(
        FixedDeposit.objects.filter(
            user=user,
            status="Active"
        ).aggregate(total=Sum("amount"))["total"] or 0
    )

    rd_total = 0

    for rd in RecurringDeposit.objects.filter(
        user=user,
        status="Active"):

        rd_total += float(rd.monthly_amount) * rd.installments_paid

    passion_total = float(
        PassionFund.objects.filter(
            user=user,
            status="Active"
        ).aggregate(total=Sum("current_amount"))["total"] or 0
    )

    existing_savings = (
        wallet_balance
        + fd_total
        + rd_total
        + passion_total
    )


    is_single = profile.marital_status == "Single"
    is_married = profile.marital_status == "Married"
    is_divorced = profile.marital_status == "Divorced"
    is_widowed = profile.marital_status == "Widowed"
    is_emergency_fund = "Emergency Fund" in profile.financial_goals
    is_higher_education = "Higher Education" in profile.financial_goals
    is_buy_house = "Buy a House" in profile.financial_goals
    is_buy_vehicle = "Buy a Vehicle" in profile.financial_goals
    is_retirement = "Retirement Savings" in profile.financial_goals
    is_vacation = "Vacation" in profile.financial_goals
    is_wealth = "Wealth Creation" in profile.financial_goals
    is_other_goal = "Other" in profile.financial_goals
    context = {
    "is_student": profile.employment_type == "Student",
    "is_salaried": profile.employment_type == "Salaried",
    "is_self_employed": profile.employment_type == "Self-Employed",
    "is_business_owner": profile.employment_type == "Business Owner",
    "is_freelancer": profile.employment_type == "Freelancer",
    "profile": profile,
    "is_single": is_single,
    "is_married": is_married,
    "receive_ai": receive_ai,
    "is_divorced": is_divorced,
    "is_widowed": is_widowed,
    "is_emergency_fund": is_emergency_fund,
    "is_higher_education": is_higher_education,
    "is_buy_house": is_buy_house,
    "is_buy_vehicle": is_buy_vehicle,
    "is_retirement": is_retirement,
    "is_vacation": is_vacation,
    "is_wealth": is_wealth,
    "is_other_goal": is_other_goal,
    "is_other": profile.employment_type == "Other",
}
    if request.method == "POST":
        profile.employment_type = request.POST.get('employment_type')
        profile.occupation = request.POST.get('occupation')
        
        # Validation and parsing of numbers
        monthly_income_raw = request.POST.get('monthly_income', '0')
        monthly_income = Decimal(monthly_income_raw) if monthly_income_raw else Decimal('0')
        
        other_monthly_income_raw = request.POST.get('other_monthly_income', '0')
        other_monthly_income = Decimal(other_monthly_income_raw) if other_monthly_income_raw else Decimal('0')
        
        monthly_savings_target_raw = request.POST.get('monthly_savings_target', '0')
        monthly_savings_target = Decimal(monthly_savings_target_raw) if monthly_savings_target_raw else Decimal('0')
        
      
        preferred_savings_percentage_raw = request.POST.get('preferred_savings_percentage', '0')
        preferred_savings_percentage = Decimal(preferred_savings_percentage_raw) if preferred_savings_percentage_raw else Decimal('0')
        
        spending_alert_limit_raw = request.POST.get('spending_alert_limit', '0')
        spending_alert_limit = Decimal(spending_alert_limit_raw) if spending_alert_limit_raw else Decimal('0')
        
        # Validations:
        # 1. Monthly Income cannot be negative
        if monthly_income < 0:
            messages.error(request, "Monthly Income cannot be negative.")
            return redirect('financial_profile')
            
           
        # 3. Savings target cannot exceed Monthly Income
        if monthly_savings_target > monthly_income:
            messages.error(request, "Monthly Savings Target cannot exceed Monthly Income.")
            return redirect('financial_profile')
            
        # 4. Preferred Savings % must be between 0 and 100
        if preferred_savings_percentage < 0 or preferred_savings_percentage > 100:
            messages.error(request, "Preferred Savings Percentage must be between 0 and 100.")
            return redirect('financial_profile')
            
        # 5. Alert Limit must be positive
        if spending_alert_limit <= 0:
            messages.error(request, "Alert Limit must be positive.")
            return redirect('financial_profile')

        profile.monthly_income = monthly_income
        profile.other_monthly_income = other_monthly_income
        profile.marital_status = request.POST.get('marital_status')
        profile.dependents = int(request.POST.get('dependents', 0))
        
        goals = request.POST.getlist('financial_goals')
        profile.financial_goals = ",".join(goals)
        profile.other_financial_goal = request.POST.get('other_financial_goal', '')
        
        profile.monthly_savings_target = monthly_savings_target

        
        profile.home_loan = Decimal(request.POST.get('home_loan', 0) or 0)
        profile.education_loan = Decimal(request.POST.get('education_loan', 0) or 0)
        profile.personal_loan = Decimal(request.POST.get('personal_loan', 0) or 0)
        profile.credit_card_debt = Decimal(request.POST.get('credit_card_debt', 0) or 0)
        profile.other_debt = Decimal(request.POST.get('other_debt', 0) or 0)
        
        profile.preferred_monthly_budget = Decimal(request.POST.get('preferred_monthly_budget', 0) or 0)
        profile.preferred_savings_percentage = preferred_savings_percentage
        profile.spending_alert_limit = spending_alert_limit
        profile.receive_ai_recommendations = request.POST.get('receive_ai_recommendations') == 'on'
        
        profile.save()
        messages.success(request, "Financial Profile Saved Successfully")
        return redirect('financial_wellness')

    selected_goals = profile.financial_goals.split(',') if profile.financial_goals else []
    
    context = {
        'user': user,
        'profile': profile,
        'selected_goals': selected_goals,
        'existing_savings': round(existing_savings, 2),
    }
    return render(request, 'financial_profile.html', context)

# ============================================================
# BENEFICIARIES
# ============================================================

def beneficiaries(request):
    """
    Display all beneficiaries belonging to the logged-in customer.
    """

    user_id = request.session.get("user_id")

    if not user_id:
        return redirect("login")

    user = get_object_or_404(
        UserAccount,
        id=user_id
    )

    beneficiary_list = Beneficiary.objects.filter(
        user=user
    ).order_by("-added_on")

    context = {
        "user": user,
        "beneficiaries": beneficiary_list
    }

    return render(
        request,
        "beneficiaries.html",
        context
    )


def add_beneficiary(request):
    """
    Add a new beneficiary for the logged-in customer.
    """

    user_id = request.session.get("user_id")

    if not user_id:
        return redirect("login")

    user = get_object_or_404(
        UserAccount,
        id=user_id
    )

    if request.method == "POST":

        print("\n" + "=" * 60)
        print("ADD BENEFICIARY")
        print("=" * 60)
        print("POST DATA:", request.POST)

        beneficiary_name = request.POST.get(
            "beneficiary_name",
            ""
        ).strip()

        beneficiary_account = request.POST.get(
            "beneficiary_account",
            ""
        ).strip()

        beneficiary_bank = request.POST.get(
            "beneficiary_bank",
            ""
        ).strip()

        ifsc_code = request.POST.get(
            "ifsc_code",
            ""
        ).strip().upper()

        print("Beneficiary Name:", beneficiary_name)
        print("Account Number:", beneficiary_account)
        print("Bank Name:", beneficiary_bank)
        print("IFSC Code:", ifsc_code)

        # Validation

        if not beneficiary_name:
            messages.error(
                request,
                "Please enter the beneficiary name."
            )
            return redirect("add_beneficiary")

        if not beneficiary_account:
            messages.error(
                request,
                "Please enter the beneficiary account number."
            )
            return redirect("add_beneficiary")

        if not beneficiary_bank:
            messages.error(
                request,
                "Please enter the bank name."
            )
            return redirect("add_beneficiary")

        if not ifsc_code:
            messages.error(
                request,
                "Please enter the IFSC code."
            )
            return redirect("add_beneficiary")

        # Check duplicate beneficiary account
        # for the same logged-in user

        existing = Beneficiary.objects.filter(
            user=user,
            beneficiary_account=beneficiary_account
        ).first()

        if existing:

            messages.error(
                request,
                "This beneficiary account is already saved."
            )

            return redirect("beneficiaries")

        # Create beneficiary using ACTUAL model fields

        Beneficiary.objects.create(
            user=user,
            beneficiary_name=beneficiary_name,
            beneficiary_account=beneficiary_account,
            beneficiary_bank=beneficiary_bank,
            ifsc_code=ifsc_code,
            trusted=False,
            verified=False
        )

        print("Beneficiary created successfully.")

        messages.success(
            request,
            f"{beneficiary_name} has been added successfully."
        )

        return redirect("beneficiaries")

    context = {
        "user": user
    }

    return render(
        request,
        "add_beneficiary.html",
        context
    )


def edit_beneficiary(request, beneficiary_id):
    """
    Edit only a beneficiary belonging to
    the logged-in customer.
    """

    user_id = request.session.get("user_id")

    if not user_id:
        return redirect("login")

    user = get_object_or_404(
        UserAccount,
        id=user_id
    )

    beneficiary = get_object_or_404(
        Beneficiary,
        id=beneficiary_id,
        user=user
    )

    if request.method == "POST":

        beneficiary_name = request.POST.get(
            "beneficiary_name",
            ""
        ).strip()

        beneficiary_account = request.POST.get(
            "beneficiary_account",
            ""
        ).strip()

        beneficiary_bank = request.POST.get(
            "beneficiary_bank",
            ""
        ).strip()

        ifsc_code = request.POST.get(
            "ifsc_code",
            ""
        ).strip().upper()

        # Validation

        if (
            not beneficiary_name
            or not beneficiary_account
            or not beneficiary_bank
            or not ifsc_code
        ):

            messages.error(
                request,
                "Please fill in all required fields."
            )

            return redirect(
                "edit_beneficiary",
                beneficiary_id=beneficiary.id
            )

        # Validate IFSC format

        import re

        if not re.match(
            r"^[A-Z]{4}0[A-Z0-9]{6}$",
            ifsc_code
        ):

            messages.error(
                request,
                "Please enter a valid IFSC code."
            )

            return redirect(
                "edit_beneficiary",
                beneficiary_id=beneficiary.id
            )

        # Check for duplicate account number
        # excluding the current beneficiary

        duplicate = Beneficiary.objects.filter(
            user=user,
            beneficiary_account=beneficiary_account
        ).exclude(
            id=beneficiary.id
        ).exists()

        if duplicate:

            messages.error(
                request,
                "Another beneficiary already uses this account number."
            )

            return redirect(
                "edit_beneficiary",
                beneficiary_id=beneficiary.id
            )

        # Update ACTUAL model fields

        beneficiary.beneficiary_name = beneficiary_name
        beneficiary.beneficiary_account = beneficiary_account
        beneficiary.beneficiary_bank = beneficiary_bank
        beneficiary.ifsc_code = ifsc_code

        beneficiary.save()

        messages.success(
            request,
            "Beneficiary updated successfully."
        )

        return redirect(
            "beneficiaries"
        )

    context = {
        "user": user,
        "beneficiary": beneficiary
    }

    return render(
        request,
        "edit_beneficiary.html",
        context
    )


def delete_beneficiary(request, beneficiary_id):
    """
    Delete only a beneficiary belonging
    to the logged-in customer.
    """

    user_id = request.session.get("user_id")

    if not user_id:
        return redirect("login")

    user = get_object_or_404(
        UserAccount,
        id=user_id
    )

    beneficiary = get_object_or_404(
        Beneficiary,
        id=beneficiary_id,
        user=user
    )

    if request.method == "POST":

        beneficiary.delete()

        messages.success(
            request,
            "Beneficiary deleted successfully."
        )

    return redirect(
        "beneficiaries"
    )

# ============================================================
# TRANSFER HISTORY
# ============================================================

def transfer_history(request):

    user_id = request.session.get("user_id")

    if not user_id:
        return redirect("login")

    user = get_object_or_404(
        UserAccount,
        id=user_id
    )

    # --------------------------------------------------------
    # DETERMINE EXISTING TRANSFER TRANSACTIONS
    # --------------------------------------------------------
    #
    # IMPORTANT:
    # Change ONLY this value if your existing Transfer view
    # uses another transaction_type.
    #

    transfers = Transaction.objects.filter(
        user=user,
        transaction_type="Transfer"
    ).order_by(
        "-created_at"
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_data = transfers.aggregate(
        total=Sum("amount"),
        average=Avg("amount"),
        largest=Max("amount")
    )

    total_transfers = total_data["total"] or 0
    average_transfer = total_data["average"] or 0
    largest_transfer = total_data["largest"] or 0

    # --------------------------------------------------------
    # THIS MONTH
    # --------------------------------------------------------

    now = timezone.localtime()

    month_start = now.replace(
        day=1,
        hour=0,
        minute=0,
        second=0,
        microsecond=0
    )

    this_month_transfers = transfers.filter(
        created_at__gte=month_start
    ).aggregate(
        total=Sum("amount")
    )["total"] or 0

    # --------------------------------------------------------
    # FILTERS
    # --------------------------------------------------------

    from_date = request.GET.get(
        "from_date",
        ""
    )

    to_date = request.GET.get(
        "to_date",
        ""
    )

    min_amount = request.GET.get(
        "min_amount",
        ""
    )

    max_amount = request.GET.get(
        "max_amount",
        ""
    )

    status = request.GET.get(
        "status",
        ""
    )

    purpose = request.GET.get(
        "purpose",
        ""
    )

    filtered_transfers = transfers

    # --------------------------------------------------------
    # DATE FILTER
    # --------------------------------------------------------

    if from_date:

        try:

            from_date_obj = datetime.strptime(
                from_date,
                "%Y-%m-%d"
            ).date()

            filtered_transfers = filtered_transfers.filter(
                created_at__date__gte=from_date_obj
            )

        except ValueError:
            pass

    if to_date:

        try:

            to_date_obj = datetime.strptime(
                to_date,
                "%Y-%m-%d"
            ).date()

            filtered_transfers = filtered_transfers.filter(
                created_at__date__lte=to_date_obj
            )

        except ValueError:
            pass

    # --------------------------------------------------------
    # AMOUNT FILTER
    # --------------------------------------------------------

    if min_amount:

        try:

            filtered_transfers = filtered_transfers.filter(
                amount__gte=float(min_amount)
            )

        except ValueError:
            pass

    if max_amount:

        try:

            filtered_transfers = filtered_transfers.filter(
                amount__lte=float(max_amount)
            )

        except ValueError:
            pass

    # --------------------------------------------------------
    # STATUS FILTER
    # --------------------------------------------------------

    if status:

        # Only apply this if Transaction has a status field.
        try:

            filtered_transfers = filtered_transfers.filter(
                status=status
            )

        except Exception:
            pass

    # --------------------------------------------------------
    # PURPOSE SEARCH
    # --------------------------------------------------------

    if purpose:

        # Try common existing fields without changing the model.
        try:

            filtered_transfers = filtered_transfers.filter(
                purpose__icontains=purpose
            )

        except Exception:

            try:

                filtered_transfers = filtered_transfers.filter(
                    description__icontains=purpose
                )

            except Exception:
                pass

    context = {
        "user": user,

        "transfers": filtered_transfers,

        "total_transfers": total_transfers,

        "this_month_transfers": this_month_transfers,

        "average_transfer": average_transfer,

        "largest_transfer": largest_transfer,

        "from_date": from_date,

        "to_date": to_date,

        "min_amount": min_amount,

        "max_amount": max_amount,

        "selected_status": status,

        "purpose": purpose
    }

    return render(
        request,
        "transfer_history.html",
        context
    )

def logout_view(request):
    """
    Logs the user out by clearing the current session.
    """
    request.session.flush()

    messages.success(request, "You have been logged out successfully.")

    return redirect('login')

def terms_view(request):

    return render(
        request,
        "terms.html"
    )

from django.shortcuts import render, redirect
from django.contrib import messages


def elder_benefits(request):

    user_id = request.session.get('user_id')

    if not user_id:
        messages.error(
            request,
            "Please log in to access Elder User Benefits."
        )
        return redirect('login')

    # Only elderly users should access this page
    if request.session.get('role') != 'elder':
        messages.error(
            request,
            "This section is available only for Elder Users."
        )
        return redirect('consumer_dashboard')

    return render(
        request,
        'elder_benefits.html'
    )

def elder_dashboard(request):

    user_id = request.session.get('user_id')

    if not user_id:
        messages.error(
            request,
            "Please log in to continue."
        )
        return redirect('login')

    user = UserAccount.objects.get(
        id=user_id
    )

    # Create wallet automatically if it does not exist
    wallet, created = Wallet.objects.get_or_create(
        user=user,
        defaults={
            "balance": 0
        }
    )

    # Recent transactions
    transactions = Transaction.objects.filter(
        user=user
    ).order_by('-created_at')[:5]

    # Due scheduled withdrawals
    due_withdrawals = ScheduledWithdrawal.objects.filter(
        user=user,
        scheduled_datetime__lte=timezone.now(),
        status='Scheduled'
    ).order_by('scheduled_datetime')

    due_withdrawals_count = due_withdrawals.count()

    single_due_withdrawal = None
    formatted_due_time = ""

    if due_withdrawals_count == 1:

        single_due_withdrawal = due_withdrawals.first()

        local_dt = timezone.localtime(
            single_due_withdrawal.scheduled_datetime
        )

        now = timezone.localtime(
            timezone.now()
        )

        time_str = local_dt.strftime(
            "%I:%M %p"
        )

        if local_dt.date() == now.date():

            formatted_due_time = (
                f"today at {time_str}"
            )

        elif local_dt.date() == (
            now.date() - timedelta(days=1)
        ):

            formatted_due_time = (
                f"yesterday at {time_str}"
            )

        else:

            formatted_due_time = (
                f"on {local_dt.strftime('%d-%b-%Y')}"
                f" at {time_str}"
            )

    # AI insights
    from analytics_engine.models import (
        AIInsight,
        AIExplanation
    )

    wellness = AIInsight.objects.filter(
        user=user,
        insight_type="Financial Wellness"
    ).first()

    debt = AIInsight.objects.filter(
        user=user,
        insight_type="Debt Risk"
    ).first()

    debt_exp = AIExplanation.objects.filter(
        user=user,
        model_name="Debt Risk"
    ).first()

    context = {

        "user": user,

        "wallet": wallet,

        "transactions": transactions,

        "due_withdrawals_count":
            due_withdrawals_count,

        "single_due_withdrawal":
            single_due_withdrawal,

        "formatted_due_time":
            formatted_due_time,

        "wellness": wellness,

        "debt": debt,

        "debt_exp": debt_exp,

    }

    return render(
        request,
        'consumer_dashboard.html',
        context
    )