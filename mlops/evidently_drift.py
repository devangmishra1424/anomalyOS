# mlops/evidently_drift.py
# Drift monitoring using Evidently AI
# Reference dataset: first 200 inference records
# Current dataset: most recent 200 records
# Run locally or triggered via "Simulate Drift" button in Analytics tab
#
# DOCUMENTED AS SIMULATED DRIFT for portfolio demonstration

import os
import json
import numpy as np
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping


LOG_PATH    = "logs/inference.jsonl"
REPORT_PATH = "reports/drift_report.html"
DRIFT_COLS  = [
    "anomaly_score",
    "calibrated_score",
    "latency_ms"
]


def load_logs(n: int = None) -> pd.DataFrame:
    if not os.path.exists(LOG_PATH):
        print(f"Log file not found: {LOG_PATH}")
        return pd.DataFrame()

    records = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    if n:
        return df.tail(n)
    return df


def simulate_drift(df: pd.DataFrame) -> pd.DataFrame:
    """
    Inject 50 OOD records to simulate distribution drift.
    DOCUMENTED AS SIMULATED everywhere — not real production drift.
    """
    ood_records = []
    for i in range(50):
        ood_records.append({
            "anomaly_score":    np.random.uniform(0.8, 1.5),
            "calibrated_score": np.random.uniform(0.8, 1.0),
            "latency_ms":       np.random.uniform(500, 2000),
            "category":         "unknown",
            "is_anomalous":     True,
            "mode":             "simulated_ood"
        })
    ood_df = pd.DataFrame(ood_records)
    return pd.concat([df, ood_df], ignore_index=True)


def run_drift_report(simulate: bool = False):
    """
    Generate Evidently drift report.
    simulate=True: inject 50 OOD records into current window.
    """
    df = load_logs()

    if len(df) < 50:
        print(f"Not enough logs for drift analysis. "
              f"Need 50+, have {len(df)}.")
        print("Run some inspections first, or use simulate=True")
        if not simulate:
            return

    # Ensure numeric columns exist
    for col in DRIFT_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Split into reference (first 200) and current (last 200)
    reference = df.head(min(200, len(df) // 2))[DRIFT_COLS]
    current   = df.tail(min(200, len(df) // 2))[DRIFT_COLS]

    if simulate:
        print("Simulating drift — injecting 50 OOD records...")
        ood_df  = simulate_drift(pd.DataFrame())[DRIFT_COLS]
        current = pd.concat([current, ood_df], ignore_index=True)

    print(f"Reference: {len(reference)} records")
    print(f"Current:   {len(current)} records")

    # Build Evidently report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    os.makedirs("reports", exist_ok=True)
    report.save_html(REPORT_PATH)

    print(f"Drift report saved: {REPORT_PATH}")
    print("NOTE: This is simulated drift for portfolio demonstration.")
    return REPORT_PATH


if __name__ == "__main__":
    import sys
    simulate = "--simulate" in sys.argv
    run_drift_report(simulate=simulate)