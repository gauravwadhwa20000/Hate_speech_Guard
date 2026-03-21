"""
Evaluation runner for Hate Speech Guardian.

Runs every test case from eval_dataset.json through the full pipeline and
scores:
  1. Verdict accuracy  — did the detector return the correct safe/unsafe label?
  2. Category match    — did the analyzer identify the right hate-speech category?
  3. Severity range    — is the severity within the expected min/max bounds?
  4. JSON validity     — did the analyzer and rewriter return parseable JSON?
  5. Rewrite quality   — did the rewriter produce a non-empty, different text?

Usage:
    python evals.py                  # run all cases
    python evals.py --ids 6 7 8     # run specific case IDs
    python evals.py --unsafe-only   # run only unsafe cases
    python evals.py --safe-only     # run only safe cases
"""

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure the project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import run_pipeline  # noqa: E402

DATASET_PATH = Path(__file__).resolve().parent / "dataset.json"

# ── Colour helpers (ANSI) ────────────────────────────────────────────────────

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _pass(msg="PASS"):
    return f"{GREEN}{msg}{RESET}"


def _fail(msg="FAIL"):
    return f"{RED}{msg}{RESET}"


def _warn(msg="WARN"):
    return f"{YELLOW}{msg}{RESET}"


# ── Evaluation logic ─────────────────────────────────────────────────────────

def evaluate_case(case: dict) -> dict:
    """Run one test case and return a scored result dict."""
    case_id = case["id"]
    text = case["text"]
    expected_verdict = case["expected_verdict"]
    expected_category = case.get("expected_category")
    min_sev = case.get("min_severity")
    max_sev = case.get("max_severity")

    print(f"\n{BOLD}━━━ Case {case_id} ━━━{RESET}")
    print(f"  Text: {text[:80]}{'...' if len(text) > 80 else ''}")

    start = time.time()
    try:
        result = run_pipeline(text)
    except Exception as e:
        elapsed = time.time() - start
        print(f"  {_fail('ERROR')}: {e}  ({elapsed:.1f}s)")
        return {
            "id": case_id,
            "error": str(e),
            "verdict_correct": False,
            "category_correct": None,
            "severity_in_range": None,
            "json_valid": False,
            "rewrite_ok": False,
            "elapsed": elapsed,
        }
    elapsed = time.time() - start

    scores = {}

    # 1. Verdict accuracy
    actual_verdict = result.get("verdict", "")
    scores["verdict_correct"] = actual_verdict == expected_verdict
    status = _pass() if scores["verdict_correct"] else _fail()
    print(f"  Verdict:  expected={expected_verdict}  actual={actual_verdict}  [{status}]")

    # For safe cases, remaining checks don't apply
    if expected_verdict == "safe":
        scores["category_correct"] = None
        scores["severity_in_range"] = None
        scores["json_valid"] = True  # no JSON expected
        scores["rewrite_ok"] = None
        print(f"  (safe case — skipping analysis/rewrite checks)")
        scores["elapsed"] = elapsed
        scores["id"] = case_id
        return scores

    # 2. Category match (fuzzy — check if expected is a substring of actual)
    actual_category = str(result.get("category", "")).lower().strip()
    if expected_category:
        scores["category_correct"] = expected_category.lower() in actual_category
        status = _pass() if scores["category_correct"] else _fail()
        print(f"  Category: expected={expected_category}  actual={actual_category}  [{status}]")
    else:
        scores["category_correct"] = None

    # 3. Severity range
    raw_sev = result.get("severity")
    try:
        actual_sev = int(raw_sev) if raw_sev is not None else None
    except (ValueError, TypeError):
        actual_sev = None

    if min_sev is not None and max_sev is not None and actual_sev is not None:
        scores["severity_in_range"] = min_sev <= actual_sev <= max_sev
        status = _pass() if scores["severity_in_range"] else _fail()
        print(f"  Severity: expected={min_sev}-{max_sev}  actual={actual_sev}  [{status}]")
    else:
        scores["severity_in_range"] = None
        print(f"  Severity: actual={actual_sev}  [{_warn('SKIP')}]")

    # 4. JSON validity (check that we got real fields, not fallback defaults everywhere)
    has_category = bool(actual_category and actual_category != "general toxicity")
    has_explanation = bool(result.get("explanation"))
    scores["json_valid"] = has_category and has_explanation
    status = _pass() if scores["json_valid"] else _warn()
    print(f"  JSON OK:  category_populated={has_category}  explanation_populated={has_explanation}  [{status}]")

    # 5. Rewrite quality
    rewrite = str(result.get("rewritten_text", "")).strip()
    rewrite_nonempty = len(rewrite) > 10
    rewrite_different = rewrite.lower() != text.lower()
    scores["rewrite_ok"] = rewrite_nonempty and rewrite_different
    status = _pass() if scores["rewrite_ok"] else _fail()
    print(f"  Rewrite:  len={len(rewrite)}  different={rewrite_different}  [{status}]")
    if rewrite_nonempty:
        print(f"            \"{rewrite[:100]}{'...' if len(rewrite) > 100 else ''}\"")

    scores["elapsed"] = elapsed
    scores["id"] = case_id
    return scores


def print_summary(results: list[dict]):
    """Print a summary table and aggregate scores."""
    total = len(results)
    errors = sum(1 for r in results if "error" in r)

    def _rate(key):
        applicable = [r for r in results if r.get(key) is not None]
        if not applicable:
            return "N/A", 0, 0
        passed = sum(1 for r in applicable if r[key])
        return f"{passed}/{len(applicable)}", passed, len(applicable)

    verdict_str, verdict_p, verdict_t = _rate("verdict_correct")
    category_str, cat_p, cat_t = _rate("category_correct")
    severity_str, sev_p, sev_t = _rate("severity_in_range")
    json_str, json_p, json_t = _rate("json_valid")
    rewrite_str, rew_p, rew_t = _rate("rewrite_ok")

    # Overall weighted score
    checks = [
        (verdict_p, verdict_t, 0.30),   # 30% weight
        (cat_p, cat_t, 0.25),           # 25% weight
        (sev_p, sev_t, 0.15),           # 15% weight
        (json_p, json_t, 0.15),         # 15% weight
        (rew_p, rew_t, 0.15),           # 15% weight
    ]
    weighted_sum = 0.0
    weight_sum = 0.0
    for passed, applicable, weight in checks:
        if applicable > 0:
            weighted_sum += (passed / applicable) * weight
            weight_sum += weight
    overall = (weighted_sum / weight_sum * 100) if weight_sum > 0 else 0

    total_time = sum(r.get("elapsed", 0) for r in results)

    print(f"\n{BOLD}{'═' * 60}{RESET}")
    print(f"{BOLD}  EVALUATION SUMMARY ({total} cases, {total_time:.0f}s total){RESET}")
    print(f"{'═' * 60}")
    print(f"  {'Metric':<25} {'Result':<12} {'Weight'}")
    print(f"  {'─' * 50}")
    print(f"  {'Verdict accuracy':<25} {verdict_str:<12} 30%")
    print(f"  {'Category match':<25} {category_str:<12} 25%")
    print(f"  {'Severity in range':<25} {severity_str:<12} 15%")
    print(f"  {'JSON validity':<25} {json_str:<12} 15%")
    print(f"  {'Rewrite quality':<25} {rewrite_str:<12} 15%")
    print(f"  {'─' * 50}")

    color = GREEN if overall >= 80 else YELLOW if overall >= 60 else RED
    print(f"  {BOLD}Overall score:{RESET}          {color}{BOLD}{overall:.1f}%{RESET}")

    if errors:
        print(f"\n  {_fail(f'{errors} case(s) had errors')}")
    print(f"{'═' * 60}\n")

    return overall


# ── CLI ───────────────────────────────────────────────────────────────────────

QUICK_EVAL_IDS = {1, 4, 6, 8, 14, 26, 31, 36, 40, 48}


def main():
    parser = argparse.ArgumentParser(description="Run evals for Hate Speech Guardian")
    parser.add_argument("--ids", nargs="+", type=int, help="Only run specific case IDs")
    parser.add_argument("--quick", action="store_true", help="Run representative 10-case subset")
    parser.add_argument("--unsafe-only", action="store_true", help="Run only unsafe cases")
    parser.add_argument("--safe-only", action="store_true", help="Run only safe cases")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    args = parser.parse_args()

    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    if args.ids:
        dataset = [c for c in dataset if c["id"] in args.ids]
    elif args.quick:
        dataset = [c for c in dataset if c["id"] in QUICK_EVAL_IDS]
    if args.unsafe_only:
        dataset = [c for c in dataset if c["expected_verdict"] == "unsafe"]
    if args.safe_only:
        dataset = [c for c in dataset if c["expected_verdict"] == "safe"]

    if not dataset:
        print("No matching test cases found.")
        return

    print(f"{CYAN}{BOLD}Hate Speech Guardian — Evaluation Run{RESET}")
    print(f"Running {len(dataset)} test case(s)...\n")

    results = []
    for case in dataset:
        result = evaluate_case(case)
        results.append(result)

    overall = print_summary(results)

    if args.save:
        out = {"overall_score": overall, "results": results}
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Results saved to {args.save}")


if __name__ == "__main__":
    main()
