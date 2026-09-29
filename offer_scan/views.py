from django.shortcuts import render, redirect
from offer_scan.utils import analyze_offer
from accounts.models import UserAccount
from .models import OfferScanHistory
def offer_scan(request):

    print("\n" + "=" * 70)
    print("ETHOSBANK AI - OFFER SCAN VIEW")
    print("=" * 70)
    if 'user_id' not in request.session:

        print(
            "[VIEW] User is not logged in."
        )

        return redirect('login')


    result = None
    error = None
    offer_text = ""


    # ---------------------------------------------------------
    # POST REQUEST
    # ---------------------------------------------------------

    if request.method == "POST":

        print(
            "\n[VIEW] POST request received."
        )

        offer_text = request.POST.get(
            "offer_text",
            ""
        )

        print(
            f"[VIEW] Received text length: "
            f"{len(offer_text)} characters"
        )


        # -----------------------------------------------------
        # EMPTY TEXT CHECK
        # -----------------------------------------------------

        if not offer_text.strip():

            print(
                "[VIEW] ERROR: Empty offer text."
            )

            error = (
                "Please enter or paste an offer "
                "before scanning."
            )


        # -----------------------------------------------------
        # AI ANALYSIS & DATABASE SAVING
        # -----------------------------------------------------

        else:

            print(
                "\n[VIEW] Sending offer to "
                "EthosBank AI inference engine..."
            )

            try:

                result = analyze_offer(
                    offer_text
                )

                print(
                    "\n[VIEW] Offer analysis completed."
                )

                # ---------------------------------------------
                # STORE OFFER SCAN IN DATABASE
                # ---------------------------------------------
                session_user_id = request.session.get('user_id')
                user_instance = None
                if session_user_id:
                    try:
                        user_instance = UserAccount.objects.filter(id=session_user_id).first()
                    except Exception as u_err:
                        print(f"[VIEW] Warning fetching UserAccount: {u_err}")

                try:
                    OfferScanHistory.objects.create(
                        user=user_instance,
                        original_text=result.get("original_text", offer_text),
                        cleaned_text=result.get("cleaned_text", ""),
                        processed_text=result.get("processed_text", ""),
                        sentiment=result.get("sentiment", ""),
                        banking_intent=result.get("banking_intent", ""),
                        ethical_classification=result.get("ethical_classification", ""),
                        ml_risk_prediction=result.get("ml_risk_prediction", ""),
                        risk_level=result.get("risk_level", "Low"),
                        risk_score=float(result.get("risk_score", 0)),
                        manipulation_score=float(result.get("manipulation_score", 0)),
                        transparency_score=float(result.get("transparency_score", 0)),
                        offer_category=result.get("offer_category", ""),
                        detected_patterns=result.get("detected_patterns", []),
                        pattern_analysis=result.get("pattern_analysis", []),
                        recommendation=result.get("recommendation", "")
                    )
                    print("[VIEW] Successfully saved scan to OfferScanHistory database table.")
                except Exception as save_err:
                    print(f"[VIEW] ERROR saving OfferScanHistory record: {save_err}")


            except Exception as e:

                print(
                    "\n[VIEW] ERROR during analysis:"
                )

                print(
                    f"[VIEW] {e}"
                )

                error = (
                    "Unable to analyze the offer. "
                    "Please try again."
                )


    # ---------------------------------------------------------
    # CONTEXT
    # ---------------------------------------------------------

    context = {

        "result": result,

        "error": error,

        "offer_text": offer_text

    }


    print(
        "\n[VIEW] Rendering offer_scan.html..."
    )


    return render(
        request,
        "offer_scan.html",
        context
    )