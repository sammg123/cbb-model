"""Model calibration analysis -- Brier Score, calibration curve.

Standalone script (not in daily cron). Run: python calibration.py
Requires settled picks in Google Sheets + predictions in clv_log.json.
"""

import json
import os
import numpy as np
from datetime import datetime

CLV_LOG_PATH = os.path.join(os.path.dirname(__file__), "clv_log.json")


def load_predictions():
    """Load all predictions from clv_log.json."""
    try:
        with open(CLV_LOG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def match_outcomes(predictions):
    """Match predictions to actual outcomes from Google Sheets.

    Returns list of (predicted_prob, actual_outcome) tuples.
    predicted_prob: the model's Sim% as a float 0-1
    actual_outcome: 1 for Hit, 0 for Miss (Pushes excluded)
    """
    from sheets import get_client, get_or_create_spreadsheet, TAB_ALL_PICKS

    client = get_client()
    sheet = get_or_create_spreadsheet(client)
    ws = sheet.worksheet(TAB_ALL_PICKS)
    all_rows = ws.get_all_records()

    # Build lookup: (date, game, bet_type, pick) -> result
    results = {}
    for row in all_rows:
        key = (row.get("Date", ""), row.get("Game", ""),
               row.get("Bet Type", ""), row.get("Pick", ""))
        results[key] = row.get("Result", "Pending")

    matched = []
    for pred in predictions:
        key = (pred.get("date", ""), pred.get("game", ""),
               pred.get("bet_type", ""), pred.get("pick", ""))
        result = results.get(key, "Pending")

        if result == "Pending":
            continue
        if result == "Push":
            continue

        # Extract predicted probability from score (EV% string like "36.8%")
        try:
            score_str = pred.get("score", "0%").replace("%", "")
            # Score is EV%, not probability -- use sim% if available
            # For now, approximate: a 55% Sim means ~55% probability
            # We'd need to store Sim% in clv_log for proper calibration
            predicted_prob = float(score_str) / 100.0
        except (ValueError, TypeError):
            continue

        actual = 1 if result == "Hit" else 0
        matched.append((predicted_prob, actual))

    return matched


def brier_score(predictions_and_outcomes):
    """Calculate Brier Score -- lower is better.

    Perfect: 0.0, Random (50/50): 0.25, Always wrong: 1.0
    """
    if not predictions_and_outcomes:
        return None

    n = len(predictions_and_outcomes)
    total = sum((p - o) ** 2 for p, o in predictions_and_outcomes)
    return total / n


def calibration_bins(predictions_and_outcomes, n_bins=5):
    """Bin predictions and compare predicted vs actual rates.

    Returns list of (bin_label, predicted_avg, actual_avg, count).
    """
    if not predictions_and_outcomes:
        return []

    # Sort by predicted probability
    sorted_po = sorted(predictions_and_outcomes, key=lambda x: x[0])

    bin_size = max(1, len(sorted_po) // n_bins)
    bins = []

    for i in range(0, len(sorted_po), bin_size):
        chunk = sorted_po[i:i + bin_size]
        if not chunk:
            continue
        pred_avg = np.mean([p for p, _ in chunk])
        actual_avg = np.mean([o for _, o in chunk])
        bins.append((f"{pred_avg:.0%}", pred_avg, actual_avg, len(chunk)))

    return bins


def print_report(predictions_and_outcomes):
    """Print calibration report."""
    if not predictions_and_outcomes:
        print("Insufficient data for calibration analysis.")
        print(f"Need settled picks with matching clv_log.json entries.")
        print(f"Currently have {len(load_predictions())} logged predictions.")
        return

    n = len(predictions_and_outcomes)
    hits = sum(o for _, o in predictions_and_outcomes)

    print(f"\n{'='*50}")
    print(f"  MODEL CALIBRATION REPORT")
    print(f"{'='*50}")
    print(f"  Settled picks analyzed: {n}")
    print(f"  Hit rate: {hits}/{n} ({hits/n*100:.1f}%)")

    bs = brier_score(predictions_and_outcomes)
    print(f"  Brier Score: {bs:.4f} (lower is better, 0.25 = random)")

    bins = calibration_bins(predictions_and_outcomes)
    if bins:
        print(f"\n  Calibration curve:")
        print(f"  {'Bin':>8s}  {'Predicted':>10s}  {'Actual':>8s}  {'Count':>6s}")
        print(f"  {'-'*36}")
        for label, pred, actual, count in bins:
            bar = "#" * int(actual * 20)
            print(f"  {label:>8s}  {pred:>10.1%}  {actual:>8.1%}  {count:>6d}  {bar}")

    print(f"\n  Note: Brier Score < 0.20 = good model")
    print(f"        Brier Score 0.20-0.25 = marginal")
    print(f"        Brier Score > 0.25 = worse than random")
    print(f"{'='*50}")


def main():
    preds = load_predictions()
    if not preds:
        print("No predictions in clv_log.json yet.")
        return

    matched = match_outcomes(preds)
    print_report(matched)


if __name__ == "__main__":
    main()
