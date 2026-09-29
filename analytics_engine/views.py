import json

from django.shortcuts import render, redirect
from django.contrib import messages

from accounts.models import UserAccount
from banking.models import Transaction
from .services import FinancialWellnessService

import plotly.graph_objects as go
import plotly.utils


def wellness_dashboard(request):
    """
    Renders the AI Financial Wellness Dashboard with metrics,
    Plotly charts, and recommendations.
    """

    user_id = request.session.get('user_id')

    if not user_id:
        messages.error(
            request,
            "Please log in to access the Wellness Dashboard."
        )
        return redirect('login')

    try:
        user = UserAccount.objects.get(id=user_id)

    except UserAccount.DoesNotExist:
        messages.error(request, "User session invalid.")
        return redirect('login')

    # ============================================================
    # EXISTING WELLNESS CALCULATION
    # ============================================================

    force_recalc = request.GET.get('refresh') == 'true'

    try:

        insights = FinancialWellnessService.get_or_calculate_insights(
            user,
            force_recalc=force_recalc
        )

    except Exception as e:

        print(f"Error calculating wellness insights: {e}")

        messages.error(
            request,
            "Could not compute wellness insights. Please verify your profile details."
        )

        insights = None

    # ============================================================
    # FALLBACK
    # ============================================================

    if not insights:

        insights = {

            'Financial Wellness': {
                'score': 0.0,
                'prediction': 'Error',
                'confidence': 0.0,
                'explanation': 'Error loading insights.'
            },

            'Debt Risk': {
                'score': 0.0,
                'prediction': 'Error',
                'confidence': 0.0,
                'explanation': 'Error loading insights.'
            },

            'Savings Forecast': {
                'score': 0.0,
                'prediction': 'Error',
                'confidence': 0.0,
                'explanation': 'Error loading insights.',
                'feature_importance': {}
            },

            'Impulse Spending': {
                'score': 0.0,
                'prediction': 'Error',
                'confidence': 0.0,
                'explanation': 'Error loading insights.',
                'feature_importance': {}
            },

            'Poverty Risk': {
                'score': 0.0,
                'prediction': 'Error',
                'confidence': 0.0,
                'explanation': 'Error loading insights.'
            },

            'Consolidated Recommendations': [
                "Please verify your profile information."
            ]
        }

    # ============================================================
    # EXISTING PLOTLY CHART VARIABLES
    # ============================================================

    savings_chart_json = "{}"
    spending_chart_json = "{}"

    # ============================================================
    # EXISTING SAVINGS FORECAST CHART
    # ============================================================

    try:

        sav_feat = insights['Savings Forecast'].get(
            'feature_importance',
            {}
        )

        if sav_feat:

            wallet_bal = sav_feat.get(
                'current_wallet_balance',
                0.0
            )

            proj_3 = sav_feat.get(
                'forecast_3_months',
                sav_feat.get(
                    '3_months_forecast',
                    wallet_bal
                )
            )

            proj_6 = sav_feat.get(
                'forecast_6_months',
                sav_feat.get(
                    '6_months_forecast',
                    wallet_bal
                )
            )

            proj_12 = sav_feat.get(
                'forecast_12_months',
                sav_feat.get(
                    '12_months_forecast',
                    wallet_bal
                )
            )

            fig_sav = go.Figure()

            fig_sav.add_trace(
                go.Scatter(
                    x=[
                        'Current',
                        '3 Months',
                        '6 Months',
                        '12 Months'
                    ],

                    y=[
                        wallet_bal,
                        proj_3,
                        proj_6,
                        proj_12
                    ],

                    mode='lines+markers',

                    line=dict(
                        color='#10B981',
                        width=3
                    ),

                    marker=dict(
                        size=10,
                        color='#0B1026',
                        line=dict(
                            color='#10B981',
                            width=2
                        )
                    ),

                    name='Projected Balance'
                )
            )

            fig_sav.update_layout(
                title=None,
                xaxis_title='Timeline',
                yaxis_title='Balance (₹)',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(
                    family="'Poppins', sans-serif",
                    size=12,
                    color="#64748B"
                ),
                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),
                height=300,

                xaxis=dict(
                    showgrid=True,
                    gridcolor='#E2E8F0'
                ),

                yaxis=dict(
                    showgrid=True,
                    gridcolor='#E2E8F0'
                )
            )

            savings_chart_json = json.dumps(
                fig_sav,
                cls=plotly.utils.PlotlyJSONEncoder
            )

    except Exception as e:

        print(
            f"Error creating savings Plotly chart: {e}"
        )

    # ============================================================
    # EXISTING SPENDING DONUT
    # ============================================================

    try:

        spend_feat = insights['Impulse Spending'].get(
            'feature_importance',
            {}
        )

        if spend_feat:

            categories = [
                'Food',
                'Shopping',
                'Travel',
                'Bills',
                'Healthcare',
                'Education',
                'Investment',
                'Subscription'
            ]

            values = []
            labels = []

            for cat in categories:

                pct = spend_feat.get(
                    f'{cat} Percentage',
                    0.0
                )

                if pct > 0:

                    values.append(pct)
                    labels.append(cat)

            if not values:

                labels = [
                    'Food',
                    'Shopping',
                    'Bills',
                    'Subscription',
                    'Travel'
                ]

                values = [
                    25.0,
                    20.0,
                    35.0,
                    10.0,
                    10.0
                ]

            fig_spend = go.Figure(
                data=[
                    go.Pie(
                        labels=labels,
                        values=values,
                        hole=.4,
                        textinfo='percent+label',

                        marker=dict(
                            colors=[
                                '#10B981',
                                '#3B82F6',
                                '#F59E0B',
                                '#EF4444',
                                '#8B5CF6',
                                '#EC4899',
                                '#6B7280',
                                '#14B8A6'
                            ]
                        )
                    )
                ]
            )

            fig_spend.update_layout(
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',

                font=dict(
                    family="'Poppins', sans-serif",
                    size=12,
                    color="#64748B"
                ),

                margin=dict(
                    l=10,
                    r=10,
                    t=10,
                    b=10
                ),

                height=300
            )

            spending_chart_json = json.dumps(
                fig_spend,
                cls=plotly.utils.PlotlyJSONEncoder
            )

    except Exception as e:

        print(
            f"Error creating spending Plotly chart: {e}"
        )

    # ============================================================
    # EXISTING SERVICE CHARTS
    # ============================================================

    try:

        charts = FinancialWellnessService.generate_charts(user)

        savings_growth_chart = charts.get(
            'savings_growth_chart',
            ''
        )

        expense_donut_chart = charts.get(
            'expense_donut_chart',
            ''
        )

    except Exception as e:

        print(
            f"Error generating Plotly charts: {e}"
        )

        savings_growth_chart = ''
        expense_donut_chart = ''

    # ============================================================
    # ============================================================
    # NEW CHARTS START HERE
    # ============================================================
    # ============================================================

    additional_charts = {
        'income_expense_chart': '',
        'monthly_spending_chart': '',
        'category_spending_chart': '',
        'savings_rate_chart': '',
        'transaction_activity_chart': '',
        'wellness_risk_chart': ''
    }

    try:

        # --------------------------------------------------------
        # Get user's transactions
        # --------------------------------------------------------

        transactions = Transaction.objects.filter(
            user=user
        ).order_by('created_at')

        # --------------------------------------------------------
        # Prepare monthly data
        # --------------------------------------------------------

        monthly_income = {}
        monthly_expense = {}
        monthly_transaction_count = {}

        category_expenses = {}

        for transaction in transactions:

            month_key = transaction.created_at.strftime(
                '%b %Y'
            )

            amount = float(transaction.amount)

            if month_key not in monthly_income:
                monthly_income[month_key] = 0.0

            if month_key not in monthly_expense:
                monthly_expense[month_key] = 0.0

            if month_key not in monthly_transaction_count:
                monthly_transaction_count[month_key] = 0

            monthly_transaction_count[month_key] += 1

            # ----------------------------------------------------
            # Deposits = income
            # Withdrawals / transfers = expense
            # ----------------------------------------------------

            if transaction.transaction_type == 'Deposit':

                monthly_income[month_key] += amount

            elif transaction.transaction_type in [
                'Withdraw',
                'Transfer'
            ]:

                monthly_expense[month_key] += amount

            # ----------------------------------------------------
            # Category spending
            # ----------------------------------------------------

            category = transaction.category or 'Other'

            if transaction.transaction_type in [
                'Withdraw',
                'Transfer'
            ]:

                category_expenses[category] = (
                    category_expenses.get(category, 0.0)
                    + amount
                )

        # --------------------------------------------------------
        # Sort months chronologically
        # --------------------------------------------------------

        months = sorted(
            set(
                list(monthly_income.keys())
                + list(monthly_expense.keys())
            ),
            key=lambda x: __import__('datetime').datetime.strptime(
                x,
                '%b %Y'
            )
        )

        # ========================================================
        # CHART 3
        # MONTHLY INCOME VS EXPENSE
        # ========================================================

        if months:

            income_values = [
                monthly_income.get(month, 0.0)
                for month in months
            ]

            expense_values = [
                monthly_expense.get(month, 0.0)
                for month in months
            ]

        else:

            months = ['No Data']

            income_values = [0]
            expense_values = [0]

        fig_income_expense = go.Figure()

        fig_income_expense.add_trace(
            go.Bar(
                x=months,
                y=income_values,
                name='Income',
                marker_color='#10B981'
            )
        )

        fig_income_expense.add_trace(
            go.Bar(
                x=months,
                y=expense_values,
                name='Expenses',
                marker_color='#EF4444'
            )
        )

        fig_income_expense.update_layout(
            title=None,
            barmode='group',
            height=320,

            xaxis_title='Month',
            yaxis_title='Amount (₹)',

            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

            font=dict(
                family="'Poppins', sans-serif",
                size=12,
                color="#64748B"
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            xaxis=dict(
                showgrid=False
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor='#E2E8F0'
            )
        )

        additional_charts[
            'income_expense_chart'
        ] = fig_income_expense.to_html(
            full_html=False,
            include_plotlyjs=False
        )

        # ========================================================
        # CHART 4
        # MONTHLY SPENDING TREND
        # ========================================================

        fig_monthly_spending = go.Figure()

        fig_monthly_spending.add_trace(
            go.Scatter(
                x=months,
                y=expense_values,

                mode='lines+markers',

                name='Monthly Spending',

                line=dict(
                    color='#F59E0B',
                    width=3
                ),

                marker=dict(
                    size=8
                ),

                fill='tozeroy'
            )
        )

        fig_monthly_spending.update_layout(
            title=None,

            height=320,

            xaxis_title='Month',
            yaxis_title='Spending (₹)',

            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

            font=dict(
                family="'Poppins', sans-serif",
                size=12,
                color="#64748B"
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            xaxis=dict(
                showgrid=False
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor='#E2E8F0'
            )
        )

        additional_charts[
            'monthly_spending_chart'
        ] = fig_monthly_spending.to_html(
            full_html=False,
            include_plotlyjs=False
        )

        # ========================================================
        # CHART 5
        # CATEGORY-WISE SPENDING
        # ========================================================

        if category_expenses:

            sorted_categories = sorted(
                category_expenses.items(),
                key=lambda x: x[1],
                reverse=True
            )

            category_names = [
                item[0]
                for item in sorted_categories
            ]

            category_values = [
                item[1]
                for item in sorted_categories
            ]

        else:

            category_names = ['No Spending Data']
            category_values = [0]

        fig_category = go.Figure()

        fig_category.add_trace(
            go.Bar(
                x=category_values,
                y=category_names,

                orientation='h',

                name='Spending',

                marker_color='#3B82F6'
            )
        )

        fig_category.update_layout(
            title=None,

            height=320,

            xaxis_title='Amount (₹)',
            yaxis_title='Category',

            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

            font=dict(
                family="'Poppins', sans-serif",
                size=12,
                color="#64748B"
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            xaxis=dict(
                showgrid=True,
                gridcolor='#E2E8F0'
            ),

            yaxis=dict(
                showgrid=False
            )
        )

        additional_charts[
            'category_spending_chart'
        ] = fig_category.to_html(
            full_html=False,
            include_plotlyjs=False
        )

        # ========================================================
        # CHART 6
        # SAVINGS RATE TREND
        # ========================================================

        savings_rates = []

        for month in months:

            income = monthly_income.get(
                month,
                0.0
            )

            expense = monthly_expense.get(
                month,
                0.0
            )

            if income > 0:

                rate = (
                    (income - expense)
                    / income
                ) * 100

            else:

                rate = 0.0

            savings_rates.append(
                round(rate, 2)
            )

        fig_savings_rate = go.Figure()

        fig_savings_rate.add_trace(
            go.Scatter(
                x=months,
                y=savings_rates,

                mode='lines+markers',

                name='Savings Rate',

                line=dict(
                    color='#8B5CF6',
                    width=3
                ),

                marker=dict(
                    size=8
                )
            )
        )

        fig_savings_rate.update_layout(
            title=None,

            height=320,

            xaxis_title='Month',
            yaxis_title='Savings Rate (%)',

            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

            font=dict(
                family="'Poppins', sans-serif",
                size=12,
                color="#64748B"
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            xaxis=dict(
                showgrid=False
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor='#E2E8F0',
                ticksuffix='%'
            )
        )

        additional_charts[
            'savings_rate_chart'
        ] = fig_savings_rate.to_html(
            full_html=False,
            include_plotlyjs=False
        )

        # ========================================================
        # CHART 7
        # TRANSACTION ACTIVITY
        # ========================================================

        transaction_counts = [
            monthly_transaction_count.get(
                month,
                0
            )
            for month in months
        ]

        fig_activity = go.Figure()

        fig_activity.add_trace(
            go.Bar(
                x=months,
                y=transaction_counts,

                name='Transactions',

                marker_color='#0EA5E9'
            )
        )

        fig_activity.update_layout(
            title=None,

            height=320,

            xaxis_title='Month',
            yaxis_title='Number of Transactions',

            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

            font=dict(
                family="'Poppins', sans-serif",
                size=12,
                color="#64748B"
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=40
            ),

            xaxis=dict(
                showgrid=False
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor='#E2E8F0'
            )
        )

        additional_charts[
            'transaction_activity_chart'
        ] = fig_activity.to_html(
            full_html=False,
            include_plotlyjs=False
        )

        # ========================================================
        # CHART 8
        # FINANCIAL WELLNESS & RISK OVERVIEW
        # ========================================================

        wellness_score = float(
            insights.get(
                'Financial Wellness',
                {}
            ).get(
                'score',
                0
            )
        )

        debt_score = float(
            insights.get(
                'Debt Risk',
                {}
            ).get(
                'score',
                0
            )
        )

        poverty_score = float(
            insights.get(
                'Poverty Risk',
                {}
            ).get(
                'score',
                0
            )
        )

        impulse_score = float(
            insights.get(
                'Impulse Spending',
                {}
            ).get(
                'score',
                0
            )
        )

        fig_wellness_risk = go.Figure()

        fig_wellness_risk.add_trace(
            go.Bar(
                x=[
                    'Financial Wellness',
                    'Debt Risk',
                    'Poverty Risk',
                    'Impulse Spending'
                ],

                y=[
                    wellness_score,
                    debt_score,
                    poverty_score,
                    impulse_score
                ],

                name='AI Score',

                marker_color=[
                    '#10B981',
                    '#EF4444',
                    '#F59E0B',
                    '#3B82F6'
                ]
            )
        )

        fig_wellness_risk.update_layout(
            title=None,

            height=320,

            xaxis_title='AI Metric',
            yaxis_title='Score',

            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',

            font=dict(
                family="'Poppins', sans-serif",
                size=12,
                color="#64748B"
            ),

            margin=dict(
                l=20,
                r=20,
                t=20,
                b=60
            ),

            xaxis=dict(
                showgrid=False
            ),

            yaxis=dict(
                showgrid=True,
                gridcolor='#E2E8F0',
                range=[0, 100]
            ),

            showlegend=False
        )

        additional_charts[
            'wellness_risk_chart'
        ] = fig_wellness_risk.to_html(
            full_html=False,
            include_plotlyjs=False
        )

    except Exception as e:

        print(
            f"Error generating additional wellness charts: {e}"
        )

    # ============================================================
    # RATING CLASS
    # ============================================================

    wellness_score = insights[
        'Financial Wellness'
    ].get(
        'score',
        50.0
    )

    rating_class = 'average'

    if wellness_score >= 80:

        rating_class = 'excellent'

    elif wellness_score >= 60:

        rating_class = 'good'

    elif wellness_score < 40:

        rating_class = 'poor'

    wellness_dashoffset = (
        502.6
        *
        (
            1
            -
            float(wellness_score)
            /
            100.0
        )
    )

    # ============================================================
    # RISK CLASSES
    # ============================================================

    debt_risk = insights[
        'Debt Risk'
    ].get(
        'prediction',
        'Low'
    ).lower()

    poverty_risk = insights[
        'Poverty Risk'
    ].get(
        'prediction',
        'Low'
    ).lower()

    # ============================================================
    # TEMPLATE INSIGHTS
    # ============================================================

    template_insights = {

        'Financial_Wellness':
            insights.get(
                'Financial Wellness',
                {}
            ),

        'Debt_Risk':
            insights.get(
                'Debt Risk',
                {}
            ),

        'Savings_Forecast':
            insights.get(
                'Savings Forecast',
                {}
            ),

        'Impulse_Spending':
            insights.get(
                'Impulse Spending',
                {}
            ),

        'Poverty_Risk':
            insights.get(
                'Poverty Risk',
                {}
            ),

        'Consolidated_Recommendations':
            insights.get(
                'Consolidated Recommendations',
                []
            )
    }

    # ============================================================
    # FINAL CONTEXT
    # ============================================================

    context = {

        'insights':
            template_insights,

        # Existing charts
        'savings_chart_json':
            savings_chart_json,

        'spending_chart_json':
            spending_chart_json,

        'savings_growth_chart':
            savings_growth_chart,

        'expense_donut_chart':
            expense_donut_chart,

        # NEW SIX CHARTS
        'income_expense_chart':
            additional_charts[
                'income_expense_chart'
            ],

        'monthly_spending_chart':
            additional_charts[
                'monthly_spending_chart'
            ],

        'category_spending_chart':
            additional_charts[
                'category_spending_chart'
            ],

        'savings_rate_chart':
            additional_charts[
                'savings_rate_chart'
            ],

        'transaction_activity_chart':
            additional_charts[
                'transaction_activity_chart'
            ],

        'wellness_risk_chart':
            additional_charts[
                'wellness_risk_chart'
            ],

        'rating_class':
            rating_class,

        'wellness_dashoffset':
            wellness_dashoffset,

        'debt_risk':
            debt_risk,

        'poverty_risk':
            poverty_risk
    }

    return render(
        request,
        'financial_wellness.html',
        context
    )