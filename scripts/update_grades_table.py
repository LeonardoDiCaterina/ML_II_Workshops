#!/usr/bin/env python3
"""
scripts/update_grades_table.py

Automated Grade & Attempt Tracker for Machine Learning II Workshops.
Runs pytest for each workshop, tracks attempts, calculates grades,
and automatically updates the progress table in README.md.
"""

import os
import sys
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime, timezone

ROOT_DIR = Path(__file__).resolve().parent.parent
TRACKER_FILE = ROOT_DIR / ".grades_tracker.json"
README_FILE = ROOT_DIR / "README.md"

WORKSHOPS = [
    ("workshop_00_revising", "00", "Foundations: Synthetic Panel, PCA/UMAP, Pipelines"),
    ("workshop_01_ts_graphics", "01", "Time Series Graphics: APIs, Datetime Indexing, ACF"),
    ("workshop_02_decomposition", "02", "Transformations, Classical & STL Decomposition"),
    ("workshop_03_benchmarks", "03", "Benchmarks, Diagnostics & Prediction Intervals"),
    ("workshop_04_evaluation", "04", "Forecast Accuracy Metrics & Walk-Forward CV"),
    ("workshop_05_regression", "05", "Time Series Regression & Spurious Correlation"),
    ("workshop_06_ets", "06", "Exponential Smoothing & ETS Model Selection"),
    ("workshop_07_arima", "07", "Stationarity, Differencing & ARIMA Identification"),
    ("workshop_08_advanced_arima", "08", "Seasonal ARIMA, SARIMAX & Fourier Terms"),
    ("workshop_09_prophet_garch", "09", "Additive Models (Prophet) & GARCH Volatility"),
    ("workshop_10_deep_learning", "10", "Sequence Modeling with PyTorch RNN & LSTM"),
    ("workshop_11_transformers", "11", "Temporal Fusion Transformer & Self-Attention"),
    ("workshop_12_hierarchical", "12", "Hierarchical Forecasting & MinT Reconciliation"),
    ("workshop_13_multivariate", "13", "Vector Autoregression (VAR) & Impulse Response"),
    ("workshop_14_practical", "14", "Production Pipelines & Automated Backtesting"),
    ("workshop_15_final_project", "15", "Capstone Tournament & Final Submission"),
]

def load_tracker():
    if TRACKER_FILE.exists():
        try:
            with open(TRACKER_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    tracker = {}
    for folder, num, topic in WORKSHOPS:
        tracker[folder] = {
            "num": num,
            "topic": topic,
            "score": None,
            "passed": 0,
            "total": 0,
            "attempts": 0,
            "status": "⚪ Pending",
            "last_run": "—"
        }
    return tracker

def save_tracker(tracker):
    with open(TRACKER_FILE, "w", encoding="utf-8") as f:
        json.dump(tracker, f, indent=2)

def run_workshop_tests(folder):
    test_dir = ROOT_DIR / folder / "tests"
    if not test_dir.exists():
        return None
    
    cmd = [sys.executable, "-m", "pytest", str(test_dir), "-q", "--tb=no"]
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    out = res.stdout + res.stderr
    
    passed_m = re.search(r"(\d+)\s+passed", out)
    failed_m = re.search(r"(\d+)\s+failed", out)
    error_m = re.search(r"(\d+)\s+error", out)
    
    passed = int(passed_m.group(1)) if passed_m else 0
    failed = int(failed_m.group(1)) if failed_m else 0
    errors = int(error_m.group(1)) if error_m else 0
    total = passed + failed + errors
    
    if total == 0:
        return None
    return passed, total

def update_all(is_template_dry_run=False):
    tracker = load_tracker()
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    for folder, num, topic in WORKSHOPS:
        w_path = ROOT_DIR / folder
        if not w_path.exists():
            continue
        
        info = tracker.get(folder, {
            "num": num, "topic": topic, "score": None,
            "passed": 0, "total": 0, "attempts": 0,
            "status": "⚪ Pending", "last_run": "—"
        })

        if not is_template_dry_run:
            test_res = run_workshop_tests(folder)
            if test_res is not None:
                passed, total = test_res
                score = int(round((passed / total) * 100)) if total > 0 else 0
                info["passed"] = passed
                info["total"] = total
                info["score"] = score
                info["attempts"] = info.get("attempts", 0) + 1
                info["last_run"] = now_str
                
                if passed == total and total > 0:
                    info["status"] = "🟢 Passed"
                elif passed > 0:
                    info["status"] = f"🟡 In Progress ({passed}/{total})"
                else:
                    info["status"] = f"🔴 Incomplete (0/{total})"
        
        tracker[folder] = info

    save_tracker(tracker)
    update_readme(tracker)

def update_readme(tracker):
    if not README_FILE.exists():
        return
    
    content = README_FILE.read_text(encoding="utf-8")
    start_tag = "<!-- AUTOGRADER_GRADES_START -->"
    end_tag = "<!-- AUTOGRADER_GRADES_END -->"

    rows = []
    rows.append("| # | Workshop | Topic Focus | Grade | Attempts | Status | Last Run |")
    rows.append("|:---:|:---|:---|:---:|:---:|:---:|:---:|")

    total_pts = 0
    max_pts = 0

    for folder, num, topic in WORKSHOPS:
        w_path = ROOT_DIR / folder
        info = tracker.get(folder, {})
        attempts = info.get("attempts", 0)
        status = info.get("status", "⚪ Pending")
        score = info.get("score")
        last_run = info.get("last_run", "—")
        
        if w_path.exists():
            folder_link = f"[`{folder}`]({folder}/)"
        else:
            folder_link = f"`{folder}`"

        if score is not None:
            grade_str = f"**{score} / 100**"
            total_pts += score
        else:
            grade_str = "—"
        max_pts += 100

        rows.append(f"| **{num}** | {folder_link} | {topic} | {grade_str} | {attempts} | {status} | {last_run} |")

    table_md = "\n".join(rows)
    summary_md = f"**Total Progress:** `{total_pts} / {max_pts} pts` across 16 workshops.\n"
    replacement = f"{start_tag}\n\n{summary_md}\n{table_md}\n\n{end_tag}"

    if start_tag in content and end_tag in content:
        pattern = re.compile(rf"{re.escape(start_tag)}.*?{re.escape(end_tag)}", re.DOTALL)
        new_content = pattern.sub(replacement, content)
    else:
        anchor = "## Instructor Commands"
        if anchor in content:
            new_content = content.replace(anchor, f"## 📊 Workshop Progress & Grade Tracker\n\n{replacement}\n\n---\n\n{anchor}")
        else:
            new_content = content + f"\n\n## 📊 Workshop Progress & Grade Tracker\n\n{replacement}\n"

    README_FILE.write_text(new_content, encoding="utf-8")
    print("✓ Updated README.md progress table successfully.")

if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    update_all(is_template_dry_run=is_dry_run)
