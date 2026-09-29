from django.urls import path
from . import views
urlpatterns = [
    path('', views.home, name='home'),
    path(
        'register/',
        views.register_view,
        name='register'
    ),
    path(
        'registration-success/',
        views.registration_success,
        name='registration_success'
    ),
    path(
    'consumer-dashboard/',
    views.consumer_dashboard,
    name='consumer_dashboard'
),
path(
    'wallet/',
    views.wallet_page,
    name='wallet'
),
path(
    'deposit/',
    views.deposit_page,
    name='deposit'
),
path(
    'withdraw/',
    views.withdraw_page,
    name='withdraw'
),
path(
    'fixed-deposit/',
    views.fixed_deposit,
    name='fixed_deposit'
),
path(
    'transfer/',
    views.transfer_page,
    name='transfer'
),
path(
    'recurring-deposit/',
    views.recurring_deposit,
    name='recurring_deposit'
),
path(
    'transactions/',
    views.transactions_page,
    name='transactions'
),
path('fixed-deposit/', views.fixed_deposit, name='fixed_deposit'),
path('recurring-deposit/', views.recurring_deposit, name='recurring_deposit'),
path('passion-fund/', views.my_passion_fund, name='passion_fund'),
path('all-deposits/', views.all_deposits, name='all_deposits'),
path(
    'pay-rd-installment/<int:rd_id>/',
    views.pay_rd_installment,
    name='pay_rd_installment'
),
path(
    'my-passion-fund/',
    views.my_passion_fund,
    name='my_passion_fund'
),

path(
    'passion-fund-topup/<int:fund_id>/',
    views.passion_fund_topup,
    name='passion_fund_topup'
),
    path(
    'login/',
    views.login_view,
    name='login'
),
    path(
    'financial-profile/',
    views.financial_profile_view,
    name='financial_profile'
),
path(
    "beneficiaries/",
    views.beneficiaries,
    name="beneficiaries"
),

path(
    "beneficiaries/add/",
    views.add_beneficiary,
    name="add_beneficiary"
),

path(
    "beneficiaries/edit/<int:beneficiary_id>/",
    views.edit_beneficiary,
    name="edit_beneficiary"
),
path("elder-dashboard/", views.elder_dashboard, name="elder_dashboard"),
path('elder-benefits/', views.elder_benefits, name='elder_benefits'),
path(
    "beneficiaries/delete/<int:beneficiary_id>/",
    views.delete_beneficiary,
    name="delete_beneficiary"
),

path(
    "transfer-history/",
    views.transfer_history,
    name="transfer_history"
),

path('terms/', views.terms_view, name='terms_view'),
]