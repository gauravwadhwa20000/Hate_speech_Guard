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

# ── Configuration ─────────────────────────────────────────────────────────────

MAX_INPUT_LENGTH = 2000          # max characters accepted per request
PIPELINE_RETRIES = 5             # retry count for rate-limited LLM calls
RETRY_BACKOFF_SECONDS = 10       # base backoff between retries
EVAL_DATASET_PATH = Path(__file__).resolve().parent / "evals" / "dataset.json"
EVAL_CACHE_PATH = Path(__file__).resolve().parent / "evals" / "cache.json"
EVAL_WORKERS = 1                 # sequential — avoids rate-limit bursts
EVAL_DELAY = 2                   # seconds between eval cases to respect rate limits

# Quick-run: representative subset (2 safe + 6 unsafe across key categories)
QUICK_EVAL_IDS = {1, 4, 6, 8, 9, 10, 14, 17}

app = Flask(__name__)

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

    return jsonify({
        "overall_score": overall,
        "metrics": metrics,
        "results": results,
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
            "error": str(e),
            "verdict_correct": False,
            "category_correct": None,
            "severity_in_range": None,
            "json_valid": False,
            "rewrite_ok": False,
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
            "verdict_correct": verdict_correct,
            "category_correct": None,
            "severity_in_range": None,
            "json_valid": True,
            "rewrite_ok": None,
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
        "verdict_correct": verdict_correct,
        "category_correct": category_correct,
        "severity_in_range": severity_ok,
        "json_valid": json_valid,
        "rewrite_ok": rewrite_ok,
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


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=5003)
