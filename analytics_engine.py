"""
╔══════════════════════════════════════════════════════════════╗
║        ExpenseTracman — Python Analytics Engine              ║
║        firebase_admin + pandas + numpy + scikit-learn        ║
╚══════════════════════════════════════════════════════════════╝

  This script is the backend analytics engine for ExpenseTracman.
  It connects to Firestore, pulls expense data for a given user,
  runs statistical analysis, detects anomalies using Isolation
  Forest (ML), and prints a full financial report to terminal.

  Usage:
      python analytics_engine.py --uid <firebase_user_id>

  Dependencies:
      pip install firebase-admin pandas numpy scikit-learn tabulate
"""

import argparse
import json
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from tabulate import tabulate
from sklearn.ensemble import IsolationForest

import firebase_admin
from firebase_admin import credentials, firestore


# ── 1. FIREBASE INIT ──────────────────────────────────────────────────────────

def init_firebase(service_account_path: str = "serviceAccountKey.json"):
    """Initialise Firebase Admin SDK using a service account JSON."""
    if not firebase_admin._apps:
        cred = credentials.Certificate(service_account_path)
        firebase_admin.initialize_app(cred)
    return firestore.client()


# ── 2. DATA FETCHING ──────────────────────────────────────────────────────────

def fetch_expenses(db, uid: str) -> pd.DataFrame:
    """
    Fetch all expense documents for a given user from Firestore.
    Returns a cleaned pandas DataFrame.
    """
    print(f"\n🔥 Connecting to Firestore...")
    docs = (
        db.collection("expenses")
          .where("uid", "==", uid)
          .order_by("createdAt", direction=firestore.Query.DESCENDING)
          .stream()
    )

    records = []
    for doc in docs:
        data = doc.to_dict()
        records.append({
            "id":            doc.id,
            "name":          data.get("name", ""),
            "amount":        float(data.get("amount", 0)),
            "category":      data.get("category", "Other"),
            "date":          data.get("date", ""),
            "paymentMethod": data.get("paymentMethod", "UPI"),
            "notes":         data.get("notes", ""),
        })

    if not records:
        print("⚠️  No expense records found for this user.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date"]      = pd.to_datetime(df["date"], errors="coerce")
    df["dayofweek"] = df["date"].dt.day_name()
    df["month"]     = df["date"].dt.to_period("M")
    df["week"]      = df["date"].dt.isocalendar().week

    print(f"✅ Fetched {len(df)} expense records.\n")
    return df


# ── 3. SUMMARY STATISTICS ─────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> dict:
    """Compute top-level KPIs from the expense DataFrame."""
    total       = df["amount"].sum()
    avg_txn     = df["amount"].mean()
    median_txn  = df["amount"].median()
    std_txn     = df["amount"].std()
    count       = len(df)
    date_range  = (df["date"].max() - df["date"].min()).days or 1
    daily_avg   = total / date_range

    return {
        "Total Spent (₹)":       round(total, 2),
        "Transactions":          count,
        "Average Transaction":   round(avg_txn, 2),
        "Median Transaction":    round(median_txn, 2),
        "Std Deviation":         round(std_txn, 2),
        "Daily Average (₹)":     round(daily_avg, 2),
        "Date Range (days)":     date_range,
    }


# ── 4. CATEGORY BREAKDOWN ─────────────────────────────────────────────────────

def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Group expenses by category and compute totals + percentages."""
    grouped = (
        df.groupby("category")["amount"]
          .agg(["sum", "count", "mean"])
          .rename(columns={"sum": "Total (₹)", "count": "Txns", "mean": "Avg (₹)"})
          .sort_values("Total (₹)", ascending=False)
          .reset_index()
    )
    total = grouped["Total (₹)"].sum()
    grouped["Share (%)"] = (grouped["Total (₹)"] / total * 100).round(1)
    grouped["Total (₹)"] = grouped["Total (₹)"].round(2)
    grouped["Avg (₹)"]   = grouped["Avg (₹)"].round(2)
    return grouped


# ── 5. MONTHLY TREND ──────────────────────────────────────────────────────────

def monthly_trend(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate spending month-over-month and compute MoM change."""
    monthly = (
        df.groupby("month")["amount"]
          .sum()
          .reset_index()
          .rename(columns={"amount": "Total (₹)"})
    )
    monthly["MoM Change (%)"] = monthly["Total (₹)"].pct_change().mul(100).round(1)
    monthly["Total (₹)"]      = monthly["Total (₹)"].round(2)
    monthly["month"]          = monthly["month"].astype(str)
    return monthly


# ── 6. DAY-OF-WEEK ANALYSIS ───────────────────────────────────────────────────

def dow_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Find which days of the week you spend the most."""
    order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    dow = (
        df.groupby("dayofweek")["amount"]
          .agg(["sum", "mean", "count"])
          .reindex(order)
          .rename(columns={"sum": "Total (₹)", "mean": "Avg (₹)", "count": "Txns"})
          .reset_index()
          .rename(columns={"dayofweek": "Day"})
    )
    dow["Total (₹)"] = dow["Total (₹)"].round(2)
    dow["Avg (₹)"]   = dow["Avg (₹)"].round(2)
    return dow


# ── 7. PAYMENT METHOD SPLIT ───────────────────────────────────────────────────

def payment_split(df: pd.DataFrame) -> pd.DataFrame:
    """Break down spending by payment method."""
    pm = (
        df.groupby("paymentMethod")["amount"]
          .agg(["sum", "count"])
          .rename(columns={"sum": "Total (₹)", "count": "Txns"})
          .sort_values("Total (₹)", ascending=False)
          .reset_index()
          .rename(columns={"paymentMethod": "Method"})
    )
    total = pm["Total (₹)"].sum()
    pm["Share (%)"]  = (pm["Total (₹)"] / total * 100).round(1)
    pm["Total (₹)"]  = pm["Total (₹)"].round(2)
    return pm


# ── 8. ANOMALY DETECTION (ML) ─────────────────────────────────────────────────

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Use scikit-learn's Isolation Forest to detect unusual transactions.
    Returns the flagged rows sorted by anomaly score (most suspicious first).

    Parameters
    ----------
    contamination : float
        Expected proportion of anomalies in the dataset (default 5%).
    """
    if len(df) < 10:
        print("⚠️  Not enough data for anomaly detection (need ≥ 10 records).\n")
        return pd.DataFrame()

    # Feature engineering: amount + encoded category
    cat_codes = pd.Categorical(df["category"]).codes
    X = np.column_stack([df["amount"].values, cat_codes])

    model = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    df = df.copy()
    df["anomaly_score"] = model.fit_predict(X)       # -1 = anomaly, 1 = normal
    df["raw_score"]     = model.score_samples(X)     # lower = more anomalous

    anomalies = (
        df[df["anomaly_score"] == -1]
          [["name", "amount", "category", "date", "raw_score"]]
          .sort_values("raw_score")
          .reset_index(drop=True)
    )
    anomalies["date"] = anomalies["date"].dt.strftime("%Y-%m-%d")
    return anomalies


# ── 9. SAVINGS RECOMMENDATION ─────────────────────────────────────────────────

def savings_tips(df: pd.DataFrame, monthly_income: float = 50000.0) -> list[str]:
    """
    Generate simple rule-based savings recommendations.

    Parameters
    ----------
    monthly_income : float
        User's approximate monthly income for ratio calculations.
    """
    tips = []
    monthly = df["amount"].sum() / max((df["date"].max() - df["date"].min()).days, 1) * 30

    # Rule 1: spending > 70% of income
    if monthly > 0.7 * monthly_income:
        tips.append(
            f"🚨 You're spending ₹{monthly:,.0f}/mo — that's "
            f"{monthly/monthly_income*100:.0f}% of your income. "
            f"Target < 60% (₹{monthly_income*0.6:,.0f})."
        )

    # Rule 2: food > 30% of total
    cat_totals = df.groupby("category")["amount"].sum()
    if "Food" in cat_totals and cat_totals["Food"] / df["amount"].sum() > 0.30:
        tips.append(
            f"🍕 Food is {cat_totals['Food']/df['amount'].sum()*100:.0f}% of spend. "
            f"Cook at home 3 days/week to cut this by ~₹{cat_totals['Food']*0.25:,.0f}."
        )

    # Rule 3: weekend spending
    weekend = df[df["dayofweek"].isin(["Saturday","Sunday"])]["amount"].sum()
    total   = df["amount"].sum()
    if total and weekend / total > 0.40:
        tips.append(
            f"📅 {weekend/total*100:.0f}% of spending happens on weekends. "
            "Set a weekend budget cap to reduce impulse buys."
        )

    # Rule 4: UPI > 80% of transactions
    upi_count = (df["paymentMethod"] == "UPI").sum()
    if upi_count / len(df) > 0.8:
        tips.append(
            "💳 80%+ of txns are UPI. Consider a cashback credit card "
            "for recurring bills to earn 1–2% back."
        )

    if not tips:
        tips.append("✅ Your spending patterns look healthy! Keep it up.")

    return tips


# ── 10. BURN RATE SCORE ───────────────────────────────────────────────────────

def burn_rate_score(df: pd.DataFrame, monthly_income: float = 50000.0) -> int:
    """
    Compute a 0–100 burn rate score. Higher = more aggressive spending.
    Combines spend ratio, std deviation, and transaction frequency.
    """
    days    = max((df["date"].max() - df["date"].min()).days, 1)
    monthly = df["amount"].sum() / days * 30
    ratio   = min(monthly / monthly_income, 1.5)          # 0–1.5 capped
    freq    = min(len(df) / days * 30 / 60, 1.0)          # txns/month normalised
    std_n   = min(df["amount"].std() / df["amount"].mean(), 2.0) / 2  # volatility

    raw   = (ratio * 0.55 + freq * 0.25 + std_n * 0.20) * 100
    score = int(min(max(raw, 0), 99))
    return score


# ── 11. PRETTY PRINT REPORT ───────────────────────────────────────────────────

def print_report(uid: str, df: pd.DataFrame):
    """Render the full analytics report to the terminal."""

    CYAN   = "\033[96m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    RED    = "\033[91m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

    def header(title):
        print(f"\n{BOLD}{CYAN}{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}{RESET}")

    print(f"\n{BOLD}{'═'*60}")
    print(f"  ExpenseTracman  ·  Python Analytics Engine v1.0")
    print(f"  User: {uid}")
    print(f"  Generated: {datetime.now().strftime('%d %b %Y  %H:%M:%S')}")
    print(f"{'═'*60}{RESET}")

    # — Summary ——————————————————————————————
    header("📊  SUMMARY KPIs")
    stats = summary_stats(df)
    for k, v in stats.items():
        print(f"  {k:<28} {GREEN}{v}{RESET}")

    # — Burn Rate ————————————————————————————
    score = burn_rate_score(df)
    color = GREEN if score < 40 else YELLOW if score < 70 else RED
    header("🔥  BURN RATE SCORE")
    bar   = "█" * (score // 5) + "░" * (20 - score // 5)
    print(f"  [{color}{bar}{RESET}]  {color}{BOLD}{score}/100{RESET}")
    label = "Low" if score < 40 else "Moderate" if score < 70 else "High"
    print(f"  Status: {color}{label} spending velocity{RESET}")

    # — Category Breakdown ———————————————————
    header("🗂️   CATEGORY BREAKDOWN")
    cat_df = category_breakdown(df)
    print(tabulate(cat_df, headers="keys", tablefmt="rounded_outline", showindex=False))

    # — Monthly Trend ————————————————————————
    header("📈  MONTHLY TREND")
    mon_df = monthly_trend(df)
    print(tabulate(mon_df, headers="keys", tablefmt="rounded_outline", showindex=False))

    # — Day-of-Week ——————————————————————————
    header("📅  SPENDING BY DAY OF WEEK")
    dow_df = dow_analysis(df)
    print(tabulate(dow_df, headers="keys", tablefmt="rounded_outline", showindex=False))

    # — Payment Methods ——————————————————————
    header("💳  PAYMENT METHOD SPLIT")
    pm_df = payment_split(df)
    print(tabulate(pm_df, headers="keys", tablefmt="rounded_outline", showindex=False))

    # — Anomalies ————————————————————————————
    header("🤖  ML ANOMALY DETECTION  (Isolation Forest)")
    anomalies = detect_anomalies(df)
    if anomalies.empty:
        print(f"  {GREEN}No anomalies detected.{RESET}")
    else:
        print(f"  {RED}{BOLD}{len(anomalies)} unusual transaction(s) flagged:{RESET}")
        print(tabulate(anomalies.drop(columns=["raw_score"]),
                       headers="keys", tablefmt="rounded_outline", showindex=False))

    # — Savings Tips —————————————————————————
    header("💡  SAVINGS RECOMMENDATIONS")
    tips = savings_tips(df)
    for tip in tips:
        print(f"  {tip}")

    print(f"\n{BOLD}{'═'*60}{RESET}\n")


# ── 12. MAIN ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ExpenseTracman Analytics Engine")
    parser.add_argument("--uid",     required=True,  help="Firebase user UID")
    parser.add_argument("--sa",      default="serviceAccountKey.json",
                                                     help="Path to service account JSON")
    parser.add_argument("--income",  type=float, default=50000.0,
                                                     help="Monthly income in ₹ (for ratios)")
    args = parser.parse_args()

    db = init_firebase(args.sa)
    df = fetch_expenses(db, args.uid)

    if df.empty:
        return

    print_report(args.uid, df)


if __name__ == "__main__":
    main()
