from django.shortcuts import render, redirect
from decimal import Decimal
from django.utils import timezone
from datetime import datetime, timedelta
from django.contrib import messages
from django.db.models import Sum, Max, Avg
from django.core.paginator import Paginator
from accounts.models import UserAccount
from .models import Wallet, Transaction, ScheduledWithdrawal

def scheduled_withdrawal_view(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')
        
    user = UserAccount.objects.get(id=user_id)
    wallet = Wallet.objects.get(user=user)
    
    # 1. Check if the process query param is passed
    if request.GET.get('process') == 'true':
        due_withdrawals = ScheduledWithdrawal.objects.filter(
            user=user,
            scheduled_datetime__lte=timezone.now(),
            status='Scheduled'
        )
        
        if not due_withdrawals.exists():
            messages.info(request, "No scheduled withdrawals are due.")
        else:
            processed_count = 0
            insufficient_funds_count = 0
            
            for w in due_withdrawals:
                wallet.refresh_from_db()
                if wallet.balance >= w.amount:
                    wallet.balance -= w.amount
                    wallet.save()
                    
                    Transaction.objects.create(
                        user=user,
                        transaction_type='Withdraw',
                        amount=w.amount,
                        category=w.purpose or 'Other',
                        merchant_name='Scheduled Withdrawal',
                        merchant_type='Banking',
                        description=f"Scheduled Withdrawal: {w.purpose}",
                        status='Completed',
                        risk_score=0.0,
                        channel='Web',
                        risk_flag=False,
                        is_recurring=False
                    )
                    
                    w.status = 'Completed'
                    w.save()
                    processed_count += 1
                else:
                    insufficient_funds_count += 1
                    
            if processed_count > 0:
                messages.success(request, f"Successfully processed {processed_count} scheduled withdrawal(s).")
            if insufficient_funds_count > 0:
                messages.error(request, f"Could not process {insufficient_funds_count} scheduled withdrawal(s) due to insufficient wallet balance.")
                
        return redirect('scheduled_withdrawal')

    # 2. Check if a normal manual process form is submitted
    if request.method == "POST" and "process_due" in request.POST:
        due_withdrawals = ScheduledWithdrawal.objects.filter(
            user=user,
            scheduled_datetime__lte=timezone.now(),
            status='Scheduled'
        )
        
        if not due_withdrawals.exists():
            messages.info(request, "No scheduled withdrawals are due.")
        else:
            processed_count = 0
            insufficient_funds_count = 0
            
            for w in due_withdrawals:
                wallet.refresh_from_db()
                if wallet.balance >= w.amount:
                    wallet.balance -= w.amount
                    wallet.save()
                    
                    Transaction.objects.create(
                        user=user,
                        transaction_type='Withdraw',
                        amount=w.amount,
                        category=w.purpose or 'Other',
                        merchant_name='Scheduled Withdrawal',
                        merchant_type='Banking',
                        description=f"Scheduled Withdrawal: {w.purpose}",
                        status='Completed',
                        risk_score=0.0,
                        channel='Web',
                        risk_flag=False,
                        is_recurring=False
                    )
                    
                    w.status = 'Completed'
                    w.save()
                    processed_count += 1
                else:
                    insufficient_funds_count += 1
                    
            if processed_count > 0:
                messages.success(request, f"Successfully processed {processed_count} scheduled withdrawal(s).")
            if insufficient_funds_count > 0:
                messages.error(request, f"Could not process {insufficient_funds_count} scheduled withdrawal(s) due to insufficient wallet balance.")
                
        return redirect('scheduled_withdrawal')

    # 3. Check scheduling form submission
    elif request.method == "POST":
        amount_str = request.POST.get('amount')
        date_str = request.POST.get('withdrawal_date')
        time_str = request.POST.get('withdrawal_time')
        purpose = request.POST.get('purpose')
        remarks = request.POST.get('remarks', '')

        try:
            amount = Decimal(amount_str)
        except (TypeError, ValueError):
            messages.error(request, "Invalid withdrawal amount.")
            return redirect('scheduled_withdrawal')

        if amount <= 100:
            messages.error(request, "Amount must be greater than ₹100.")
            return redirect('scheduled_withdrawal')

        if amount > wallet.balance:
            messages.error(request, "Amount cannot exceed wallet balance.")
            return redirect('scheduled_withdrawal')

        try:
            datetime_str = f"{date_str} {time_str}"
            try:
                naive_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                naive_dt = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
                
            scheduled_dt = timezone.make_aware(naive_dt, timezone.get_current_timezone())
        except Exception:
            messages.error(request, "Invalid date or time format.")
            return redirect('scheduled_withdrawal')

        current_time = timezone.now()
        current_date = timezone.localtime(current_time).date()
        scheduled_local_date = timezone.localtime(scheduled_dt).date()
        
        if scheduled_local_date < current_date:
            messages.error(request, "Date cannot be in the past.")
            return redirect('scheduled_withdrawal')
        elif scheduled_dt < current_time:
            messages.error(request, "Scheduled time cannot be in the past.")
            return redirect('scheduled_withdrawal')

        ScheduledWithdrawal.objects.create(
            user=user,
            amount=amount,
            scheduled_datetime=scheduled_dt,
            purpose=purpose,
            remarks=remarks,
            status='Scheduled'
        )
        
        messages.success(request, "Withdrawal scheduled successfully.")
        return redirect('scheduled_withdrawal')

    # Fetch withdrawals
    upcoming_withdrawals = ScheduledWithdrawal.objects.filter(
        user=user,
        status='Scheduled'
    ).order_by('scheduled_datetime')
    
    past_withdrawals = ScheduledWithdrawal.objects.filter(
        user=user
    ).exclude(status='Scheduled').order_by('-scheduled_datetime')

    context = {
        'user': user,
        'wallet': wallet,
        'upcoming_withdrawals': upcoming_withdrawals,
        'past_withdrawals': past_withdrawals,
    }
    
    return render(request, 'scheduled_withdrawal.html', context)

def cancel_withdrawal(request, withdrawal_id):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')
        
    try:
        w = ScheduledWithdrawal.objects.get(id=withdrawal_id, user_id=user_id)
        if w.status == 'Scheduled':
            w.status = 'Cancelled'
            w.save()
            messages.success(request, "Scheduled withdrawal cancelled successfully.")
        else:
            messages.error(request, "This withdrawal cannot be cancelled.")
    except ScheduledWithdrawal.DoesNotExist:
        messages.error(request, "Scheduled withdrawal not found.")
        
    return redirect('scheduled_withdrawal')

def withdrawal_history_view(request):
    user_id = request.session.get('user_id')
    if not user_id:
        return redirect('login')
        
    user = UserAccount.objects.get(id=user_id)
    
    # Completed withdrawals are Transaction objects with transaction_type='Withdraw'
    completed_txns = Transaction.objects.filter(user=user, transaction_type='Withdraw')
    
    # 1. Top Statistics (based on COMPLETED/processed withdrawals)
    total_withdrawals = completed_txns.aggregate(Sum('amount'))['amount__sum'] or Decimal('0.00')
    largest_withdrawal = completed_txns.aggregate(Max('amount'))['amount__max'] or Decimal('0.00')
    avg_withdrawal = completed_txns.aggregate(Avg('amount'))['amount__avg'] or Decimal('0.00')
    
    # This month's withdrawals
    now = timezone.now()
    this_month_start = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    this_month_withdrawals = completed_txns.filter(created_at__gte=this_month_start).aggregate(Sum('amount'))['amount__sum'] or Decimal('0.00')
    
    # 2. Get active & cancelled scheduled withdrawals
    scheduled_qs = ScheduledWithdrawal.objects.filter(user=user).exclude(status='Completed')
    
    # 3. Apply Filters
    start_date_str = request.GET.get('start_date')
    if start_date_str:
        try:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
            completed_txns = completed_txns.filter(created_at__date__gte=start_date)
            scheduled_qs = scheduled_qs.filter(scheduled_datetime__date__gte=start_date)
        except ValueError:
            pass

    end_date_str = request.GET.get('end_date')
    if end_date_str:
        try:
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
            completed_txns = completed_txns.filter(created_at__date__lte=end_date)
            scheduled_qs = scheduled_qs.filter(scheduled_datetime__date__lte=end_date)
        except ValueError:
            pass

    min_amount_str = request.GET.get('min_amount')
    if min_amount_str:
        try:
            min_amount = Decimal(min_amount_str)
            completed_txns = completed_txns.filter(amount__gte=min_amount)
            scheduled_qs = scheduled_qs.filter(amount__gte=min_amount)
        except ValueError:
            pass

    max_amount_str = request.GET.get('max_amount')
    if max_amount_str:
        try:
            max_amount = Decimal(max_amount_str)
            completed_txns = completed_txns.filter(amount__lte=max_amount)
            scheduled_qs = scheduled_qs.filter(amount__lte=max_amount)
        except ValueError:
            pass

    purpose = request.GET.get('purpose')
    if purpose:
        completed_txns = completed_txns.filter(category__icontains=purpose)
        scheduled_qs = scheduled_qs.filter(purpose__icontains=purpose)

    status_filter = request.GET.get('status')
    if not status_filter:
        status_filter = 'All'

    # Combine in memory
    withdrawals_list = []
    
    # Add Completed from Transaction
    if status_filter in ['All', 'Completed']:
        for t in completed_txns:
            withdrawals_list.append({
                'id': f"TXN-{t.id:06d}",
                'raw_id': t.id,
                'type': 'Transaction',
                'date': t.created_at,
                'amount': t.amount,
                'purpose': t.category,
                'status': 'Completed',
                'channel': t.channel,
                'remarks': t.description
            })
            
    # Add Scheduled / Cancelled from ScheduledWithdrawal
    if status_filter in ['All', 'Scheduled', 'Cancelled']:
        filtered_scheduled = scheduled_qs
        if status_filter in ['Scheduled', 'Cancelled']:
            filtered_scheduled = scheduled_qs.filter(status=status_filter)
            
        for w in filtered_scheduled:
            withdrawals_list.append({
                'id': f"SCH-{w.id:06d}",
                'raw_id': w.id,
                'type': 'ScheduledWithdrawal',
                'date': w.scheduled_datetime,
                'amount': w.amount,
                'purpose': w.purpose,
                'status': w.status,
                'channel': 'Web',
                'remarks': w.remarks or ''
            })

    # Sort in memory by date descending
    withdrawals_list.sort(key=lambda x: x['date'], reverse=True)

    # 4. Pagination
    paginator = Paginator(withdrawals_list, 10)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    context = {
        'user': user,
        'page_obj': page_obj,
        'total_withdrawals': total_withdrawals,
        'this_month_withdrawals': this_month_withdrawals,
        'avg_withdrawal': avg_withdrawal,
        'largest_withdrawal': largest_withdrawal,
        'start_date': start_date_str or '',
        'end_date': end_date_str or '',
        'min_amount': min_amount_str or '',
        'max_amount': max_amount_str or '',
        'status_filter': status_filter,
        'purpose': purpose or '',
    }
    
    return render(request, 'withdrawal_history.html', context)

