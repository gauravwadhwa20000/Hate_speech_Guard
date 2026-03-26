import json
import logging
import re
import threading
import time
from pathlib import Path

from flask import Flask, request, jsonify, render_template
from crewai import Crew
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from agents import hate_speech_detector, hate_speech_analyzer, content_rewriter
from tasks import create_detection_task, create_analysis_task, create_rewrite_task
from keep_alive import start_keep_alive

# ── Configuration ─────────────────────────────────────────────────────────────

MAX_INPUT_LENGTH = 2000          # max characters accepted per request
PIPELINE_RETRIES = 5             # retry count for rate-limited LLM calls
RETRY_BACKOFF_SECONDS = 10       # base backoff between retries
EVAL_DATASET_PATH = Path(__file__).resolve().parent / "evals" / "dataset.json"
EVAL_CACHE_PATH = Path(__file__).resolve().parent / "evals" / "cache.json"
EVAL_WORKERS = 1                 # sequential — avoids rate-limit bursts
EVAL_DELAY = 2                   # seconds between eval cases to respect rate limits

# Quick-run: representative subset across key scenarios
QUICK_EVAL_IDS = {1, 4, 6, 8, 14, 26, 31, 36, 40, 48}

app = Flask(__name__)

# Start background keep-alive pinger (prevents Render free-tier sleep)
start_keep_alive()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)


def parse_json_output(raw, defaults):
    """Parse a JSON object from agent output, with fallback defaults."""
    text = str(raw).strip()
    # Strip markdown code fences if the model wraps output
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return dict(defaults)
        else:
            return dict(defaults)
    return {k: data.get(k, v) for k, v in defaults.items()}


log = logging.getLogger(__name__)


def run_pipeline(text):
    """Run the 3-agent pipeline."""
    def _kickoff_with_retry(crew, inputs, retries=PIPELINE_RETRIES):
        for attempt in range(retries):
            try:
                return crew.kickoff(inputs=inputs)
            except Exception as e:
                if "429" in str(e) and attempt < retries - 1:
                    wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                    log.warning("Rate-limited (429), retrying in %ds (attempt %d/%d)", wait, attempt + 1, retries)
                    time.sleep(wait)
                else:
                    raise

    # Step 1: Detection
    log.info("Pipeline started — detecting hate speech")
    detect_task = create_detection_task(text)
    detect_crew = Crew(agents=[hate_speech_detector], tasks=[detect_task], verbose=False)
    detect_result = str(_kickoff_with_retry(detect_crew, {"text": text})).strip().lower()

    is_unsafe = "unsafe" in detect_result or "hate" in detect_result

    if not is_unsafe:
        log.info("Verdict: safe")
        return {
            "verdict": "safe",
            "category": None,
            "severity": None,
            "toxic_words": None,
            "explanation": None,
            "rewritten_text": None,
        }

    # Step 2: Analysis
    log.info("Verdict: unsafe — running analysis")
    analyze_task = create_analysis_task(text)
    analyze_crew = Crew(agents=[hate_speech_analyzer], tasks=[analyze_task], verbose=False)
    analyze_result = str(_kickoff_with_retry(analyze_crew, {"text": text}))
    analysis = parse_json_output(analyze_result, {
        "category": "general toxicity",
        "sub_category": "",
        "target_group": "",
        "severity": 5,
        "intent": "",
        "confidence": 5,
        "toxic_words": "",
        "explanation": "",
    })

    # Step 3: Rewrite
    log.info("Analysis complete — rewriting")
    rewrite_task = create_rewrite_task(text)
    rewrite_crew = Crew(agents=[content_rewriter], tasks=[rewrite_task], verbose=False)
    rewrite_result = str(_kickoff_with_retry(rewrite_crew, {"text": text}))
    rewrite = parse_json_output(rewrite_result, {"rewritten_text": ""})

    log.info("Pipeline complete — category=%s severity=%s", analysis["category"], analysis["severity"])
    return {
        "verdict": "unsafe",
        "category": analysis["category"],
        "sub_category": analysis["sub_category"],
        "target_group": analysis["target_group"],
        "severity": analysis["severity"],
        "intent": analysis["intent"],
        "confidence": analysis["confidence"],
        "toxic_words": analysis["toxic_words"],
        "explanation": analysis["explanation"],
        "rewritten_text": rewrite["rewritten_text"],
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text", "").strip()
    if not text:
        return jsonify({"error": "Please enter some text to analyze."}), 400
    if len(text) > MAX_INPUT_LENGTH:
        return jsonify({"error": f"Text too long (max {MAX_INPUT_LENGTH} characters)."}), 400
    try:
        result = run_pipeline(text)
        return jsonify(result)
    except Exception as e:
        log.exception("Pipeline error")
        return jsonify({"error": "Analysis failed. Please try again."}), 500


# ── Eval progress tracking ───────────────────────────────────────────────────

_eval_progress = {"done": 0, "total": 0, "current_text": ""}
_eval_lock = threading.Lock()


@app.route("/eval-progress")
def eval_progress():
    with _eval_lock:
        return jsonify(dict(_eval_progress))


@app.route("/run-evals", methods=["POST"])
def run_evals():
    filter_type = request.form.get("filter", "all")
    ids_raw = request.form.get("ids", "")
    use_cache = request.form.get("use_cache", "true") == "true"

    with open(EVAL_DATASET_PATH) as f:
        dataset = json.load(f)

    if ids_raw:
        ids = {int(x) for x in ids_raw.split(",") if x.strip().isdigit()}
        dataset = [c for c in dataset if c["id"] in ids]
        use_cache = False  # always rerun explicit IDs
    elif filter_type == "quick":
        dataset = [c for c in dataset if c["id"] in QUICK_EVAL_IDS]
    elif filter_type == "safe":
        dataset = [c for c in dataset if c["expected_verdict"] == "safe"]
    elif filter_type == "unsafe":
        dataset = [c for c in dataset if c["expected_verdict"] == "unsafe"]

    # Sort: safe cases first (1 API call each) then unsafe (3 calls each)
    dataset.sort(key=lambda c: 0 if c["expected_verdict"] == "safe" else 1)

    # Load disk cache
    eval_cache = {}
    if use_cache and EVAL_CACHE_PATH.exists():
        try:
            eval_cache = {r["id"]: r for r in json.loads(EVAL_CACHE_PATH.read_text())}
        except (json.JSONDecodeError, KeyError):
            eval_cache = {}

    with _eval_lock:
        _eval_progress["done"] = 0
        _eval_progress["total"] = len(dataset)
        _eval_progress["current_text"] = ""

    results = []
    for i, case in enumerate(dataset):
        # Use cached result if available
        if use_cache and case["id"] in eval_cache:
            cached = eval_cache[case["id"]]
            cached["elapsed"] = 0  # mark as cached
            results.append(cached)
            with _eval_lock:
                _eval_progress["done"] = len(results)
            log.info("Eval case %d: using cached result", case["id"])
            continue

        with _eval_lock:
            _eval_progress["current_text"] = case["text"]

        r = _evaluate_single(case)
        results.append(r)

        with _eval_lock:
            _eval_progress["done"] = len(results)

        # Rate-limit delay between API calls (skip after last case)
        if i < len(dataset) - 1:
            time.sleep(EVAL_DELAY)

    # Sort results by case ID for consistent ordering
    results.sort(key=lambda r: r["id"])

    # Save results to disk cache (merge with existing)
    try:
        existing = {}
        if EVAL_CACHE_PATH.exists():
            existing = {r["id"]: r for r in json.loads(EVAL_CACHE_PATH.read_text())}
        for r in results:
            if r.get("elapsed", 1) > 0:  # don't overwrite with cached (elapsed=0)
                existing[r["id"]] = r
        EVAL_CACHE_PATH.write_text(json.dumps(list(existing.values()), indent=2))
        log.info("Eval cache saved (%d entries)", len(existing))
    except Exception:
        log.warning("Failed to save eval cache", exc_info=True)

    # Compute aggregate metrics
    metrics = _compute_metrics(results)
    overall = _compute_overall(metrics)
    agent_reports = _compute_agent_reports(results)
    deep_analysis = _compute_deep_analysis(results)

    return jsonify({
        "overall_score": overall,
        "metrics": metrics,
        "results": results,
        "agent_reports": agent_reports,
        "deep_analysis": deep_analysis,
    })


def _evaluate_single(case):
    """Evaluate one test case through the pipeline."""
    start = time.time()
    try:
        result = run_pipeline(case["text"])
    except Exception as e:
        return {
            "id": case["id"],
            "text": case["text"],
            "expected_verdict": case["expected_verdict"],
            "actual_verdict": None,
            "error": str(e),
            "verdict_correct": False,
            "category_correct": None,
            "severity_in_range": None,
            "json_valid": False,
            "rewrite_ok": False,
            "actual_category": None,
            "expected_category": case.get("expected_category"),
            "actual_severity": None,
            "expected_severity_range": None,
            "actual_rewrite": None,
            "scenario": case.get("scenario", "unknown"),
            "notes": case.get("notes", ""),
            "elapsed": time.time() - start,
        }

    elapsed = time.time() - start
    expected = case["expected_verdict"]
    actual = result.get("verdict", "")

    verdict_correct = actual == expected

    if expected == "safe":
        return {
            "id": case["id"],
            "text": case["text"],
            "expected_verdict": expected,
            "actual_verdict": actual,
            "verdict_correct": verdict_correct,
            "category_correct": None,
            "severity_in_range": None,
            "json_valid": True,
            "rewrite_ok": None,
            "actual_category": None,
            "expected_category": None,
            "actual_severity": None,
            "expected_severity_range": None,
            "actual_rewrite": None,
            "scenario": case.get("scenario", "unknown"),
            "notes": case.get("notes", ""),
            "elapsed": elapsed,
        }

    # Category match (fuzzy)
    actual_cat = str(result.get("category", "")).lower().strip()
    exp_cat = case.get("expected_category")
    category_correct = (exp_cat.lower() in actual_cat) if exp_cat else None

    # Severity range
    raw_sev = result.get("severity")
    try:
        actual_sev = int(raw_sev) if raw_sev is not None else None
    except (ValueError, TypeError):
        actual_sev = None
    min_s, max_s = case.get("min_severity"), case.get("max_severity")
    severity_ok = (min_s <= actual_sev <= max_s) if (min_s is not None and max_s is not None and actual_sev is not None) else None

    # JSON validity
    has_cat = bool(actual_cat and actual_cat != "general toxicity")
    has_expl = bool(result.get("explanation"))
    json_valid = has_cat and has_expl

    # Rewrite quality
    rewrite = str(result.get("rewritten_text", "")).strip()
    rewrite_ok = len(rewrite) > 10 and rewrite.lower() != case["text"].lower()

    return {
        "id": case["id"],
        "text": case["text"],
        "expected_verdict": expected,
        "actual_verdict": actual,
        "verdict_correct": verdict_correct,
        "category_correct": category_correct,
        "severity_in_range": severity_ok,
        "json_valid": json_valid,
        "rewrite_ok": rewrite_ok,
        "actual_category": actual_cat or None,
        "expected_category": exp_cat,
        "actual_severity": actual_sev,
        "expected_severity_range": f"{min_s}-{max_s}" if min_s is not None else None,
        "actual_rewrite": rewrite[:200] if rewrite else None,
        "scenario": case.get("scenario", "unknown"),
        "notes": case.get("notes", ""),
        "elapsed": elapsed,
    }


def _compute_metrics(results):
    """Compute pass/total for each metric."""
    def _count(key):
        applicable = [r for r in results if r.get(key) is not None]
        passed = sum(1 for r in applicable if r[key])
        return passed, len(applicable)

    vp, vt = _count("verdict_correct")
    cp, ct = _count("category_correct")
    sp, st = _count("severity_in_range")
    jp, jt = _count("json_valid")
    rp, rt = _count("rewrite_ok")

    return {
        "verdict_passed": vp, "verdict_total": vt,
        "category_passed": cp, "category_total": ct,
        "severity_passed": sp, "severity_total": st,
        "json_passed": jp, "json_total": jt,
        "rewrite_passed": rp, "rewrite_total": rt,
    }


def _compute_overall(m):
    """Weighted overall score."""
    checks = [
        (m["verdict_passed"], m["verdict_total"], 0.30),
        (m["category_passed"], m["category_total"], 0.25),
        (m["severity_passed"], m["severity_total"], 0.15),
        (m["json_passed"], m["json_total"], 0.15),
        (m["rewrite_passed"], m["rewrite_total"], 0.15),
    ]
    ws = wt = 0.0
    for p, t, w in checks:
        if t > 0:
            ws += (p / t) * w
            wt += w
    return (ws / wt * 100) if wt > 0 else 0


def _compute_agent_reports(results):
    """Per-agent performance breakdown."""

    # ── Agent 1: Hate Speech Detector (verdict) ──
    all_cases = [r for r in results if "error" not in r or not r.get("error")]
    error_cases = [r for r in results if r.get("error")]
    total = len(results)
    correct = sum(1 for r in results if r.get("verdict_correct"))
    false_pos = [r for r in results if r.get("expected_verdict") == "safe" and not r.get("verdict_correct")]
    false_neg = [r for r in results if r.get("expected_verdict") == "unsafe" and not r.get("verdict_correct")]
    agent1 = {
        "name": "Agent 1 — Hate Speech Detector",
        "role": "Binary classification: safe vs unsafe",
        "accuracy": round(correct / total * 100, 1) if total else 0,
        "passed": correct,
        "total": total,
        "false_positives": len(false_pos),
        "false_negatives": len(false_neg),
        "errors": len(error_cases),
        "failed_cases": [
            {"id": r["id"], "text": r["text"][:80], "expected": r["expected_verdict"],
             "issue": "False Positive (flagged safe text)" if r["expected_verdict"] == "safe"
                      else "False Negative (missed hate speech)"}
            for r in false_pos + false_neg
        ],
    }

    # ── Agent 2: Content Analyzer (category, severity, json) ──
    cat_applicable = [r for r in results if r.get("category_correct") is not None]
    cat_passed = sum(1 for r in cat_applicable if r["category_correct"])
    sev_applicable = [r for r in results if r.get("severity_in_range") is not None]
    sev_passed = sum(1 for r in sev_applicable if r["severity_in_range"])
    json_applicable = [r for r in results if r.get("json_valid") is not None]
    json_passed = sum(1 for r in json_applicable if r["json_valid"])

    a2_failed = []
    for r in results:
        issues = []
        if r.get("category_correct") is False:
            issues.append("Wrong category")
        if r.get("severity_in_range") is False:
            issues.append("Severity out of range")
        if r.get("json_valid") is False and r.get("expected_verdict") == "unsafe":
            issues.append("Invalid/incomplete JSON")
        if issues:
            a2_failed.append({"id": r["id"], "text": r["text"][:80], "issue": "; ".join(issues)})

    agent2 = {
        "name": "Agent 2 — Content Analyzer",
        "role": "Categorization, severity scoring, structured output",
        "category_accuracy": round(cat_passed / len(cat_applicable) * 100, 1) if cat_applicable else None,
        "category_passed": cat_passed,
        "category_total": len(cat_applicable),
        "severity_accuracy": round(sev_passed / len(sev_applicable) * 100, 1) if sev_applicable else None,
        "severity_passed": sev_passed,
        "severity_total": len(sev_applicable),
        "json_accuracy": round(json_passed / len(json_applicable) * 100, 1) if json_applicable else None,
        "json_passed": json_passed,
        "json_total": len(json_applicable),
        "failed_cases": a2_failed,
    }

    # ── Agent 3: Content Rewriter ──
    rw_applicable = [r for r in results if r.get("rewrite_ok") is not None]
    rw_passed = sum(1 for r in rw_applicable if r["rewrite_ok"])

    a3_failed = [
        {"id": r["id"], "text": r["text"][:80], "issue": "Rewrite too short or identical"}
        for r in rw_applicable if not r["rewrite_ok"]
    ]

    agent3 = {
        "name": "Agent 3 — Content Rewriter",
        "role": "Generate respectful alternative text",
        "accuracy": round(rw_passed / len(rw_applicable) * 100, 1) if rw_applicable else None,
        "passed": rw_passed,
        "total": len(rw_applicable),
        "failed_cases": a3_failed,
    }

    return [agent1, agent2, agent3]


def _compute_deep_analysis(results):
    """Generate per-agent actionable analysis to improve instructions and backstory."""

    # Precompute failure sets
    false_positives = [r for r in results if r.get("expected_verdict") == "safe" and not r.get("verdict_correct")]
    false_negatives = [r for r in results if r.get("expected_verdict") == "unsafe" and not r.get("verdict_correct")]
    category_mismatches = [r for r in results if r.get("category_correct") is False]
    severity_misses = [r for r in results if r.get("severity_in_range") is False]
    json_failures = [r for r in results if r.get("json_valid") is False and r.get("expected_verdict") == "unsafe"]
    rewrite_failures = [r for r in results if r.get("rewrite_ok") is False]
    error_cases = [r for r in results if r.get("error")]

    total = len(results)
    total_safe = len([r for r in results if r.get("expected_verdict") == "safe"])
    total_unsafe = len([r for r in results if r.get("expected_verdict") == "unsafe"])

    # Group failures by scenario tag
    def _scenario_breakdown(case_list):
        buckets = {}
        for r in case_list:
            scen = r.get("scenario", "unknown") if isinstance(r, dict) else "unknown"
            if scen not in buckets:
                buckets[scen] = []
            buckets[scen].append(r)
        return buckets

    # ═══════════════════════════════════════════════════════════════════════
    # AGENT 1 — Hate Speech Detector
    # ═══════════════════════════════════════════════════════════════════════
    det_correct = sum(1 for r in results if r.get("verdict_correct"))
    det_accuracy = round(det_correct / total * 100, 1) if total else 0

    det_failure_details = []
    for r in false_positives:
        det_failure_details.append({
            "case_id": r["id"],
            "type": "FALSE_POSITIVE",
            "text": r["text"][:120],
            "scenario": r.get("scenario", "unknown"),
            "diagnosis": (
                f"Flagged safe text as unsafe. The detector's 'when in doubt, classify as unsafe' "
                f"rule may be too aggressive for this type of content ({r.get('scenario', 'unknown')})."
            ),
            "suggested_rule": (
                f"Add safe-text recognition for '{r.get('scenario', 'unknown')}' scenarios: "
                "policy criticism, civic opinions, cultural appreciation, and balanced statements "
                "about sensitive topics are SAFE."
            ),
        })
    for r in false_negatives:
        exp_cat = r.get("expected_category") or "unknown"
        det_failure_details.append({
            "case_id": r["id"],
            "type": "FALSE_NEGATIVE",
            "text": r["text"][:120],
            "scenario": r.get("scenario", "unknown"),
            "expected_category": exp_cat,
            "diagnosis": (
                f"Missed {exp_cat} ({r.get('scenario', 'unknown')} scenario). "
                f"The detector failed to recognize this hate speech pattern."
            ),
            "suggested_rule": (
                f"Add detection rule for '{r.get('scenario', 'unknown')}' pattern: "
                f"'{exp_cat.upper()}: {r.get('notes', 'recognize this pattern')}'."
            ),
        })

    # Weakness by scenario type
    fp_scenarios = _scenario_breakdown(false_positives)
    fn_scenarios = _scenario_breakdown(false_negatives)

    det_scenario_analysis = []
    for scen, cases in fn_scenarios.items():
        det_scenario_analysis.append({
            "scenario": scen,
            "direction": "MISSED",
            "count": len(cases),
            "case_ids": [c["id"] for c in cases],
            "insight": f"Detector missed {len(cases)} case(s) in '{scen}' scenario — needs better pattern recognition for this type.",
        })
    for scen, cases in fp_scenarios.items():
        det_scenario_analysis.append({
            "scenario": scen,
            "direction": "OVER_FLAGGED",
            "count": len(cases),
            "case_ids": [c["id"] for c in cases],
            "insight": f"Detector over-flagged {len(cases)} safe case(s) in '{scen}' scenario — needs safe-content recognition for this type.",
        })

    det_backstory_suggestions = []
    det_task_suggestions = []

    if false_negatives:
        missed_cats = list(set(r.get("expected_category", "unknown") for r in false_negatives))
        missed_scenarios = list(set(r.get("scenario", "unknown") for r in false_negatives))
        det_backstory_suggestions.append(
            f"You are especially vigilant about detecting: {', '.join(missed_cats)}. "
            f"You never miss these patterns even when they use coded language, sarcasm, "
            f"microaggressions, or pseudoscientific framing."
        )
        for scen in missed_scenarios:
            scen_cases = fn_scenarios[scen]
            if scen == "coded-language":
                det_task_suggestions.append(
                    "CODED LANGUAGE: Phrases like 'I'm not racist but...', 'if you know what I mean', "
                    "'those people', 'stay in their own neighborhoods' are ALWAYS unsafe even without explicit slurs."
                )
            elif scen == "microaggression":
                det_task_suggestions.append(
                    "MICROAGGRESSIONS: Backhanded compliments ('so articulate for...'), "
                    "questioning belonging ('where are you actually from?'), and professional gatekeeping "
                    "('you don't look like an engineer') are unsafe."
                )
            elif scen == "sarcastic-hate":
                det_task_suggestions.append(
                    "SARCASTIC HATE: When sarcasm or irony is used to demean a group "
                    "(e.g., 'Oh sure, THOSE people are so civilized'), classify as unsafe."
                )
            elif scen == "pseudoscience":
                det_task_suggestions.append(
                    "PSEUDOSCIENCE: Claims framed as 'biology', 'studies show', or 'it's just a fact' "
                    "that assign traits/roles by race or gender are ALWAYS unsafe."
                )
            elif scen == "casual-hate":
                det_task_suggestions.append(
                    "CASUAL HATE: 'Lol typical behavior', 'no offense but', "
                    "and similar minimizing preambles followed by stereotyping are unsafe."
                )
            elif scen == "incitement":
                det_task_suggestions.append(
                    "INCITEMENT: 'Better off without', 'teach them a lesson', 'round them up' "
                    "and similar eliminationist/violent rhetoric are always unsafe (severity 8+)."
                )
            elif scen == "multi-target":
                det_task_suggestions.append(
                    "MULTI-TARGET: Text combining slurs from multiple categories "
                    "(e.g., xenophobia + sexism) is always unsafe."
                )

    if false_positives:
        safe_scenarios = list(set(r.get("scenario", "unknown") for r in false_positives))
        det_backstory_suggestions.append(
            "You understand that policy criticism, civic opinions, cultural appreciation, "
            "diversity support, and balanced statements are NOT hate speech, even when they "
            "mention sensitive topics like immigration, crime statistics, or cultural practices."
        )
        det_task_suggestions.append(
            "SAFE PATTERNS — classify as 'safe':\n"
            "- Policy/government criticism (not targeting a people group)\n"
            "- Cultural appreciation or positive diversity statements\n"
            "- Balanced/nuanced views on sensitive topics\n"
            "- Advocacy for rights or social programs\n"
            "- Empathetic statements about affected communities"
        )

    agent1_analysis = {
        "agent_name": "Agent 1 — Hate Speech Detector",
        "agent_file": "agents.py → hate_speech_detector",
        "task_file": "tasks.py → create_detection_task()",
        "accuracy": det_accuracy,
        "total_cases": total,
        "passed": det_correct,
        "false_positives": len(false_positives),
        "false_negatives": len(false_negatives),
        "errors": len(error_cases),
        "failure_details": det_failure_details,
        "scenario_analysis": det_scenario_analysis,
        "backstory_suggestions": det_backstory_suggestions,
        "task_prompt_suggestions": det_task_suggestions,
        "strength_areas": [],
        "weakness_areas": [],
    }

    # Compute strength/weakness by scenario
    all_scenarios = set(r.get("scenario", "unknown") for r in results)
    for scen in sorted(all_scenarios):
        scen_cases = [r for r in results if r.get("scenario") == scen]
        scen_correct = sum(1 for r in scen_cases if r.get("verdict_correct"))
        if len(scen_cases) > 0:
            scen_acc = round(scen_correct / len(scen_cases) * 100, 1)
            entry = {"scenario": scen, "accuracy": scen_acc, "total": len(scen_cases), "passed": scen_correct}
            if scen_acc >= 100:
                agent1_analysis["strength_areas"].append(entry)
            elif scen_acc < 100:
                agent1_analysis["weakness_areas"].append(entry)

    # ═══════════════════════════════════════════════════════════════════════
    # AGENT 2 — Content Analyzer
    # ═══════════════════════════════════════════════════════════════════════
    cat_applicable = [r for r in results if r.get("category_correct") is not None]
    cat_passed = sum(1 for r in cat_applicable if r["category_correct"])
    sev_applicable = [r for r in results if r.get("severity_in_range") is not None]
    sev_passed = sum(1 for r in sev_applicable if r["severity_in_range"])
    json_applicable = [r for r in results if r.get("json_valid") is not None and r.get("expected_verdict") == "unsafe"]
    json_pass = sum(1 for r in json_applicable if r["json_valid"])

    # Category confusion matrix
    confusion_pairs = {}
    for r in category_mismatches:
        exp = r.get("expected_category", "unknown").lower()
        act = (r.get("actual_category") or "unknown").lower()
        pair_key = f"{exp} -> {act}"
        if pair_key not in confusion_pairs:
            confusion_pairs[pair_key] = {"expected": exp, "actual": act, "cases": []}
        confusion_pairs[pair_key]["cases"].append({
            "id": r["id"], "text": r["text"][:80], "scenario": r.get("scenario", "unknown"),
        })

    category_confusion = []
    for key, data in confusion_pairs.items():
        category_confusion.append({
            "expected": data["expected"],
            "actual": data["actual"],
            "count": len(data["cases"]),
            "case_ids": [c["id"] for c in data["cases"]],
            "fix": (
                f"Add rule: 'When targeting based on {data['expected']}-related traits, "
                f"classify as \"{data['expected']}\", NOT \"{data['actual']}\".' "
                f"Key differentiator: look at the target group and specific language used."
            ),
        })

    # Severity drift
    over_rated = []
    under_rated = []
    for r in severity_misses:
        actual_sev = r.get("actual_severity")
        exp_range = r.get("expected_severity_range", "")
        if actual_sev is not None and exp_range:
            try:
                min_s, max_s = map(int, exp_range.split("-"))
            except (ValueError, TypeError):
                continue
            entry = {
                "id": r["id"], "text": r["text"][:80],
                "actual": actual_sev, "expected_range": exp_range,
                "scenario": r.get("scenario", "unknown"),
                "category": r.get("expected_category", "unknown"),
            }
            if actual_sev > max_s:
                entry["drift"] = actual_sev - max_s
                over_rated.append(entry)
            elif actual_sev < min_s:
                entry["drift"] = min_s - actual_sev
                under_rated.append(entry)

    severity_drift = []
    if over_rated:
        avg = round(sum(x["drift"] for x in over_rated) / len(over_rated), 1)
        severity_drift.append({
            "direction": "OVER_RATING",
            "count": len(over_rated),
            "avg_drift": avg,
            "cases": over_rated,
            "fix": (
                "Add calibration: 'Stereotyping without threats = 3-5. Only assign 7+ for "
                "dehumanization, explicit slurs, or incitement. 9-10 is reserved for direct "
                "threats of violence or genocide.'"
            ),
        })
    if under_rated:
        avg = round(sum(x["drift"] for x in under_rated) / len(under_rated), 1)
        severity_drift.append({
            "direction": "UNDER_RATING",
            "count": len(under_rated),
            "avg_drift": avg,
            "cases": under_rated,
            "fix": (
                "Add severity floors: 'Dehumanization is always 7+. Calls for violence/exclusion "
                "are always 8+. Pseudoscientific racism is always 7+. Eliminationist rhetoric is 9+.'"
            ),
        })

    ana_failure_details = []
    for r in category_mismatches:
        ana_failure_details.append({
            "case_id": r["id"],
            "type": "WRONG_CATEGORY",
            "text": r["text"][:120],
            "expected": r.get("expected_category"),
            "actual": r.get("actual_category"),
            "scenario": r.get("scenario", "unknown"),
        })
    for r in severity_misses:
        ana_failure_details.append({
            "case_id": r["id"],
            "type": "SEVERITY_MISS",
            "text": r["text"][:120],
            "expected_range": r.get("expected_severity_range"),
            "actual": r.get("actual_severity"),
            "scenario": r.get("scenario", "unknown"),
        })
    for r in json_failures:
        ana_failure_details.append({
            "case_id": r["id"],
            "type": "JSON_INVALID",
            "text": r["text"][:120],
            "scenario": r.get("scenario", "unknown"),
        })

    ana_backstory_suggestions = []
    ana_task_suggestions = []

    if category_mismatches:
        ana_backstory_suggestions.append(
            "When classifying, pay close attention to these distinctions:\n" +
            "\n".join(
                f"- '{d['expected']}' vs '{d['actual']}': look at the target group and specific language"
                for d in category_confusion
            )
        )
        ana_task_suggestions.append(
            "CATEGORY DISAMBIGUATION RULES:\n" +
            "\n".join(
                f"- If targeting based on {d['expected']}-related traits → '{d['expected']}', NOT '{d['actual']}'"
                for d in category_confusion
            )
        )
    if severity_drift:
        ana_backstory_suggestions.append(
            "Calibrate severity carefully: microaggressions=1-3, stereotyping/bias=3-5, "
            "explicit slurs/generalizations=5-7, dehumanization=7-8, threats/incitement=8-10."
        )
        ana_task_suggestions.append(
            "SEVERITY CALIBRATION:\n"
            "- Microaggressions, backhanded compliments: 1-3\n"
            "- Stereotyping, casual bias, coded language: 3-5\n"
            "- Explicit slurs, hateful generalizations: 5-7\n"
            "- Dehumanization, severe degradation: 7-8\n"
            "- Direct threats, calls for violence/genocide: 8-10\n"
            "- Eliminationist rhetoric: 9-10"
        )
    if json_failures:
        ana_backstory_suggestions.append(
            "Always output a single valid JSON object with ALL required fields populated. "
            "Never wrap in markdown code fences. Never leave fields empty."
        )
        ana_task_suggestions.append(
            "CRITICAL: Response must be ONLY a raw JSON object. No markdown, "
            "no code fences, no explanation text. All fields must be non-empty."
        )

    ana_total_checks = len(cat_applicable) + len(sev_applicable) + len(json_applicable)
    ana_total_passed = cat_passed + sev_passed + json_pass
    ana_accuracy = round(ana_total_passed / ana_total_checks * 100, 1) if ana_total_checks else 0

    agent2_analysis = {
        "agent_name": "Agent 2 — Content Analyzer",
        "agent_file": "agents.py → hate_speech_analyzer",
        "task_file": "tasks.py → create_analysis_task()",
        "accuracy": ana_accuracy,
        "category_accuracy": round(cat_passed / len(cat_applicable) * 100, 1) if cat_applicable else None,
        "category_passed": cat_passed,
        "category_total": len(cat_applicable),
        "severity_accuracy": round(sev_passed / len(sev_applicable) * 100, 1) if sev_applicable else None,
        "severity_passed": sev_passed,
        "severity_total": len(sev_applicable),
        "json_accuracy": round(json_pass / len(json_applicable) * 100, 1) if json_applicable else None,
        "json_passed": json_pass,
        "json_total": len(json_applicable),
        "failure_details": ana_failure_details,
        "category_confusion": category_confusion,
        "severity_drift": severity_drift,
        "backstory_suggestions": ana_backstory_suggestions,
        "task_prompt_suggestions": ana_task_suggestions,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # AGENT 3 — Content Rewriter
    # ═══════════════════════════════════════════════════════════════════════
    rw_applicable = [r for r in results if r.get("rewrite_ok") is not None]
    rw_passed = sum(1 for r in rw_applicable if r["rewrite_ok"])
    rw_accuracy = round(rw_passed / len(rw_applicable) * 100, 1) if rw_applicable else None

    rew_failure_details = []
    for r in rw_applicable:
        if not r["rewrite_ok"]:
            actual_rw = r.get("actual_rewrite", "")
            if not actual_rw or len(actual_rw.strip()) < 10:
                issue = "EMPTY_OR_TOO_SHORT"
                diagnosis = "Rewrite was empty or under 10 characters — agent may have returned raw JSON parse failure or no content."
            elif actual_rw.strip().lower() == r["text"].strip().lower():
                issue = "IDENTICAL_TO_ORIGINAL"
                diagnosis = "Rewrite was identical to the original hateful text — agent failed to transform the content."
            else:
                issue = "LOW_QUALITY"
                diagnosis = "Rewrite was produced but didn't pass quality checks."

            rew_failure_details.append({
                "case_id": r["id"],
                "type": issue,
                "text": r["text"][:120],
                "actual_rewrite": (actual_rw[:150] if actual_rw else None),
                "scenario": r.get("scenario", "unknown"),
                "diagnosis": diagnosis,
            })

    # Group rewrite failures by scenario
    rew_scenario_failures = _scenario_breakdown([r for r in rw_applicable if not r["rewrite_ok"]])

    rew_backstory_suggestions = []
    rew_task_suggestions = []

    if rew_failure_details:
        empty_count = sum(1 for f in rew_failure_details if f["type"] == "EMPTY_OR_TOO_SHORT")
        identical_count = sum(1 for f in rew_failure_details if f["type"] == "IDENTICAL_TO_ORIGINAL")

        if empty_count > 0:
            rew_backstory_suggestions.append(
                "You ALWAYS produce a rewrite of at least 2 full sentences. You never return "
                "an empty response or just the JSON key without meaningful content."
            )
            rew_task_suggestions.append(
                "Your rewritten_text must be at least 2 complete sentences (minimum 20 words). "
                "If you struggle to rewrite, focus on the speaker's underlying topic and express "
                "it constructively."
            )
        if identical_count > 0:
            rew_backstory_suggestions.append(
                "Your rewrite must be substantially different from the original. The original is "
                "hateful — your version must remove ALL hatred while keeping the topic."
            )
            rew_task_suggestions.append(
                "NEVER return the original text as your rewrite. The rewrite MUST remove all "
                "hateful content and be clearly different from the input."
            )

        # Scenario-specific suggestions
        for scen, cases in rew_scenario_failures.items():
            if scen == "microaggression":
                rew_task_suggestions.append(
                    "For MICROAGGRESSIONS: Rewrite to express genuine interest or respect "
                    "without the underlying assumption (e.g., 'You're so articulate' → "
                    "'I really enjoyed your well-reasoned argument')."
                )
            elif scen == "coded-language":
                rew_task_suggestions.append(
                    "For CODED LANGUAGE: Identify the actual concern underneath the coded "
                    "speech and express it directly and respectfully."
                )

    agent3_analysis = {
        "agent_name": "Agent 3 — Content Rewriter",
        "agent_file": "agents.py → content_rewriter",
        "task_file": "tasks.py → create_rewrite_task()",
        "accuracy": rw_accuracy,
        "total": len(rw_applicable),
        "passed": rw_passed,
        "failure_details": rew_failure_details,
        "scenario_failures": dict(
            (k, len(v)) for k, v in rew_scenario_failures.items()
        ),
        "backstory_suggestions": rew_backstory_suggestions,
        "task_prompt_suggestions": rew_task_suggestions,
    }

    # ═══════════════════════════════════════════════════════════════════════
    # OVERALL SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    total_failures = (
        len(false_positives) + len(false_negatives) + len(category_mismatches) +
        len(severity_misses) + len(json_failures) + len(rewrite_failures) + len(error_cases)
    )

    if total_failures == 0:
        summary = (
            f"All {total} eval cases passed across all 3 agents. Instructions and backstory "
            "are well-calibrated. Consider adding more edge cases (microaggressions, coded language, "
            "intersectional hate) to stress-test further."
        )
    else:
        issues_list = []
        if false_negatives:
            issues_list.append(f"Agent 1 missed {len(false_negatives)} hate speech case(s)")
        if false_positives:
            issues_list.append(f"Agent 1 over-flagged {len(false_positives)} safe case(s)")
        if category_mismatches:
            issues_list.append(f"Agent 2 misclassified {len(category_mismatches)} category(ies)")
        if severity_misses:
            issues_list.append(f"Agent 2 miscalibrated severity in {len(severity_misses)} case(s)")
        if json_failures:
            issues_list.append(f"Agent 2 had {len(json_failures)} JSON output failure(s)")
        if rewrite_failures:
            issues_list.append(f"Agent 3 had {len(rewrite_failures)} rewrite failure(s)")
        if error_cases:
            issues_list.append(f"{len(error_cases)} pipeline error(s)")

        summary = (
            f"Found {total_failures} issue(s) across {total} cases. "
            f"{'; '.join(issues_list)}. "
            "Apply the per-agent backstory and task prompt suggestions below to improve accuracy."
        )

    return {
        "summary": summary,
        "total_cases": total,
        "total_safe": total_safe,
        "total_unsafe": total_unsafe,
        "total_failures": total_failures,
        "agent1": agent1_analysis,
        "agent2": agent2_analysis,
        "agent3": agent3_analysis,
    }


# ── Dataset Management Endpoints ──────────────────────────────────────────────

@app.route("/dataset", methods=["GET"])
def get_dataset():
    """Return the full eval dataset."""
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/dataset/<int:case_id>", methods=["PUT"])
def update_dataset_case(case_id):
    """Update a single test case by ID."""
    data = request.get_json(force=True)
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    for i, case in enumerate(dataset):
        if case["id"] == case_id:
            # Only allow updating known fields
            for key in ("text", "expected_verdict", "expected_category",
                        "min_severity", "max_severity", "notes", "scenario"):
                if key in data:
                    val = data[key]
                    # Coerce severity fields to int or None
                    if key in ("min_severity", "max_severity"):
                        val = int(val) if val is not None and str(val).strip() != "" else None
                    dataset[i][key] = val
            EVAL_DATASET_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
            log.info("Updated dataset case %d", case_id)
            return jsonify({"ok": True, "case": dataset[i]})

    return jsonify({"ok": False, "error": f"Case {case_id} not found"}), 404


@app.route("/dataset", methods=["POST"])
def add_dataset_case():
    """Add a new test case to the dataset."""
    data = request.get_json(force=True)
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"ok": False, "error": "Text is required"}), 400

    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    new_id = max((c["id"] for c in dataset), default=0) + 1
    new_case = {
        "id": new_id,
        "text": text,
        "expected_verdict": data.get("expected_verdict", "unsafe"),
        "expected_category": data.get("expected_category") or None,
        "min_severity": int(data["min_severity"]) if data.get("min_severity") else None,
        "max_severity": int(data["max_severity"]) if data.get("max_severity") else None,
        "notes": data.get("notes", ""),
        "scenario": data.get("scenario", ""),
    }
    dataset.append(new_case)
    EVAL_DATASET_PATH.write_text(json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Added dataset case %d", new_id)
    return jsonify({"ok": True, "case": new_case})


@app.route("/dataset/<int:case_id>", methods=["DELETE"])
def delete_dataset_case(case_id):
    """Delete a test case by ID."""
    with open(EVAL_DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    new_dataset = [c for c in dataset if c["id"] != case_id]
    if len(new_dataset) == len(dataset):
        return jsonify({"ok": False, "error": f"Case {case_id} not found"}), 404

    EVAL_DATASET_PATH.write_text(json.dumps(new_dataset, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Deleted dataset case %d", case_id)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5000)
