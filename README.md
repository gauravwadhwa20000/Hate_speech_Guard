# Hate Speech Guardian

A multi-agent AI content moderation pipeline built with **CrewAI** and **Flask**. Three specialized AI agents collaborate in sequence to **detect**, **analyze**, and **rewrite** hate speech in user-submitted text — all powered by free LLMs accessed through **OpenRouter**.

The goal of this project is to demonstrate how multiple AI agents, each with a distinct role, can be orchestrated into a cooperative pipeline to tackle a real-world problem: **online hate speech moderation** — with a built-in evaluation framework that tells you exactly how to improve each agent.

---

## Project Overview

Online platforms struggle with content moderation at scale. Manual review is slow and expensive; a single LLM prompt can miss nuance. **Hate Speech Guardian** solves this by breaking the moderation task into three focused stages, each handled by a dedicated AI agent:

1. **Detection** — A safety-focused agent decides whether the text is **safe** or **unsafe**.
2. **Analysis** — If unsafe, an analysis agent returns a strict JSON object with the hate category, severity (1–10), intent, target group, confidence score, and the exact toxic words.
3. **Rewriting** — A creative agent returns a JSON object containing the hateful text rewritten as a respectful, constructive alternative while preserving the speaker's underlying concern.

The agents communicate through a **CrewAI sequential pipeline**, and results are displayed in a modern Flask web UI with animated progress indicators.

On top of the pipeline, the project includes a **comprehensive evaluation framework** with 50 labeled test cases, **per-agent deep analysis** that generates copy-ready fix suggestions, a **live dataset editor**, and a **downloadable report** — turning "it seems to work" into measurable, improvable scores.

---

## How It Works

```
User enters text in the Web UI
           │
           ▼
┌──────────────────────────┐
│  Agent 1 — Detector      │  arcee-ai/trinity-large-preview (free)
│  Role: Content Safety     │  Classifies input as "safe" or "unsafe"
└────────────┬─────────────┘
             │ unsafe?
             ▼
┌──────────────────────────┐
│  Agent 2 — Analyzer      │  arcee-ai/trinity-large-preview (free)
│  Role: Sociolinguistics   │  Category, severity, intent, toxic words, explanation
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│  Agent 3 — Rewriter      │  arcee-ai/trinity-large-preview (free)
│  Role: Content Moderator  │  Rewrites text into a respectful version
└────────────┬─────────────┘
             │
             ▼
     JSON response → Web UI
```

If the Detector classifies the text as **safe**, the pipeline **short-circuits** and returns immediately — no analysis or rewrite is performed.

---

## Model

All three agents use the same free model from **Arcee AI**, accessed through **OpenRouter**:

| Agent | Model | Provider | Why This Model? |
|-------|-------|----------|-----------------|
| Detector | `arcee-ai/trinity-large-preview:free` | Arcee AI | Fast inference, strong instruction-following, handles binary classification well |
| Analyzer | `arcee-ai/trinity-large-preview:free` | Arcee AI | Good at structured JSON output and following exact field formats |
| Rewriter | `arcee-ai/trinity-large-preview:free` | Arcee AI | Produces clean, respectful rewrites while preserving speaker intent |

The model is hosted on OpenRouter and accessed through their unified API. The `:free` suffix means it costs **$0** — no billing required.

> **Note:** The architecture supports different models per agent. To use specialized models, simply change the model string in `agents.py` (e.g., swap the Detector to `nvidia/nemotron-nano-9b-v2:free` for faster binary classification).

---

## OpenRouter

**OpenRouter** is a unified API gateway that gives access to **200+ LLMs** from providers like OpenAI, Meta, Google, NVIDIA, Mistral, and many more — all through a single API key and a consistent API format (compatible with the OpenAI SDK).

### Why OpenRouter?

- **Single API key** — Access models from dozens of providers without managing multiple accounts.
- **Free models** — Many models are available at no cost (suffixed with `:free`), perfect for prototyping and demos.
- **Easy switching** — Change the model by editing a single string (e.g., swap `arcee-ai/trinity-large-preview:free` for `nvidia/nemotron-nano-9b-v2:free`).
- **CrewAI integration** — CrewAI's `LLM` class supports OpenRouter out of the box using the `openrouter/` prefix.

### How It's Used in This Project

In `agents.py`, each agent is initialized with an OpenRouter model:

```python
from crewai.llm import LLM

detector_llm = LLM(
    model="openrouter/arcee-ai/trinity-large-preview:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=128,
)
```

The `openrouter/` prefix tells CrewAI (via **LiteLLM**) to route the request through the OpenRouter API. CrewAI uses LiteLLM under the hood for non-native providers — it must be installed separately (`pip install litellm`). The API key is stored in a `.env` file and loaded via `python-dotenv`.

---

## Features

### Pipeline Features
- **3-Agent Pipeline** — Demonstrates CrewAI multi-agent orchestration with sequential task execution
- **Hate Speech Detection** — Binary safe/unsafe classification with 11 detailed detection rules (slurs, stereotyping, coded language, sarcasm, pseudoscience, incitement, etc.)
- **Category Classification** — Identifies type: racism, sexism, xenophobia, religious hatred, homophobia, transphobia, ableism, ageism, classism, body shaming, and more
- **Severity Rating** — Visual severity bar from 1 (mild microaggression) to 10 (threats of violence)
- **Intent Classification** — Distinguishes deliberate hatred from ignorant, casual, or satirical speech
- **Confidence Score** — The analyzer reports how confident it is in its own assessment (1–10)
- **Toxic Word Highlighting** — Pinpoints the exact offensive words/phrases from the input
- **Content Rewriting** — Transforms hateful text into respectful alternatives while preserving the speaker's underlying concern
- **Retry with Backoff** — Automatic retry (up to 5 attempts) with exponential backoff for rate-limited API calls (HTTP 429)
- **Modern Web UI** — Clean dark glassmorphism theme with animated agent progress steps

### Evaluation Features
- **50 Labeled Test Cases** — Across 10 hate speech scenarios (explicit hate, coded language, microaggressions, pseudoscience, sarcastic hate, incitement, and more)
- **Weighted Scoring** — 5 metrics (verdict 30%, category 25%, severity 15%, JSON validity 15%, rewrite quality 15%)
- **Per-Agent Deep Analysis** — After evals, get a diagnosis of each agent's failures with copy-ready fix suggestions for both `agents.py` (backstory) and `tasks.py` (task prompts)
- **Download Report** — Save the full eval results, per-agent analysis, and fix suggestions as a `.txt` file
- **Live Dataset Editor** — View, search, add, edit, and delete test cases directly in the web UI
- **Disk Caching** — Already-scored eval results are cached to `evals/cache.json`; reruns skip completed cases
- **Rerun Failures** — Re-test only the cases that failed, keeping passing results cached
- **Quick Run** — Test 10 representative cases instead of all 50

---

## Severity Calibration

When Agent 2 (Analyzer) assigns a severity score, it follows this rubric:

| Score | Level | Description | Examples |
|-------|-------|-------------|----------|
| **1–2** | Mild | Microaggressions, thoughtless remarks | "You're so articulate for a..." |
| **3–4** | Moderate | Clear bias, stereotyping, casual discrimination | "Women just aren't as logical" |
| **5–6** | High | Explicit hate speech, slurs, derogatory generalizations | Racial slurs, "all X are criminals" |
| **7–8** | Severe | Dehumanization, incitement to discrimination | Comparing people to animals |
| **9–10** | Extreme | Direct threats of violence, calls for genocide | "They should all be..." |

Each test case in the dataset has a `min_severity` and `max_severity` range. The eval system checks if Agent 2's score falls within that range:
- **Too high** = over-rating (e.g., rating a microaggression as severity 8)
- **Too low** = under-rating / severity drift (e.g., rating explicit hate as severity 3)

---

## Detection Rules (11 Categories)

The Detector (Agent 1) checks the input against these 11 specific rules. Text matching **any** of them is classified as `unsafe`:

| # | Rule | What It Catches |
|---|------|-----------------|
| 1 | **Explicit Hate** | Slurs, racial epithets, derogatory terms for any group |
| 2 | **Generalizations** | Sweeping negative claims ("all X are Y", "X people always...") |
| 3 | **Dehumanization** | Comparing a group to animals, diseases, vermin, or objects |
| 4 | **Stereotyping** | Portraying an entire group as criminal, inferior, dangerous, or immoral |
| 5 | **Threats & Incitement** | Calls for violence, exclusion, or harm against a group |
| 6 | **Coded Language & Dog Whistles** | Euphemisms like "go back to your country", dog whistles |
| 7 | **Sarcastic Hate** | Irony or sarcasm masking hateful intent ("Oh sure, THOSE people...") |
| 8 | **Implied Discrimination** | Suggesting a group doesn't belong or is inherently lesser |
| 9 | **Class-Based Hate** | Targeting people based on economic status, poverty, homelessness |
| 10 | **Ageism / Ableism** | Mocking or degrading based on age or disability |
| 11 | **Pseudoscientific Racism** | Claiming racial differences in intelligence or ability as "facts" or "science" |

Text is classified as `safe` **only** if it is genuinely respectful, neutral, or constructive with no hateful undertones.

---

## Category Taxonomy

Agent 2 classifies hate speech into one of these primary categories:

| Category | Description |
|----------|-------------|
| `racism` | Targeting based on race or ethnicity — includes dehumanization, pseudoscientific claims |
| `sexism` | Targeting based on gender — includes misogyny, gender-role stereotyping |
| `xenophobia` | Targeting based on national origin or immigration status |
| `religious hatred` | Targeting based on religion or religious practices |
| `homophobia` | Targeting based on sexual orientation |
| `transphobia` | Targeting based on gender identity |
| `ableism` | Targeting based on disability |
| `ageism` | Targeting based on age |
| `classism` | Targeting based on economic status or social class |
| `body shaming` | Targeting based on physical appearance |
| `general toxicity` | Hateful content that doesn't fit a specific category above |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| AI Framework | [CrewAI](https://github.com/crewAIInc/crewAI) (multi-agent orchestration) |
| LLM Routing | [LiteLLM](https://github.com/BerriAI/litellm) (enables CrewAI → OpenRouter) |
| LLM Gateway | [OpenRouter](https://openrouter.ai) (unified access to 200+ models) |
| Model | `arcee-ai/trinity-large-preview:free` (all 3 agents) |
| Web Framework | Flask |
| Language | Python 3.11+ |
| Frontend | Single-page HTML/CSS/JS with glassmorphism dark theme |

---

## Project Structure

```
hate-speech-guardian/
├── app.py                  # Flask web server + pipeline orchestration + eval engine + dataset API
├── agents.py               # 3 CrewAI agents (Detector, Analyzer, Rewriter)
├── tasks.py                # Task definitions with detection rules, JSON schemas, severity rubric
├── evals/
│   ├── runner.py           # CLI evaluation runner — scores pipeline against test cases
│   ├── dataset.json        # 50 labeled test cases (16 safe + 34 unsafe, 10 scenarios)
│   └── cache.json          # Disk cache for eval results (auto-generated)
├── templates/
│   └── index.html          # Web UI (Analyze + Evals + Dataset tabs)
├── .env.example            # Environment variable template — copy to .env
├── requirements.txt        # Python dependencies
├── test_models.py          # Model connectivity test script
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## Quick Start

> **Requires Python 3.11 or 3.12.** CrewAI is not compatible with Python 3.14. Check with `python --version`.

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/hate-speech-guardian.git
cd hate-speech-guardian
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
.\venv\Scripts\Activate.ps1     # Windows PowerShell
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

```bash
cp .env.example .env            # Linux/Mac
copy .env.example .env          # Windows
```

Edit `.env` and add your [OpenRouter API key](https://openrouter.ai/keys) (free, no credit card needed):

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 5. Run the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Usage

The UI has **three tabs**:

### Analyze Tab

1. Enter any text in the input box
2. Click **Analyze Text**
3. Watch the 3 agents process sequentially:
   - **Agent 1** detects if the text is safe or contains hate speech
   - **Agent 2** classifies the category, rates severity, and highlights toxic words
   - **Agent 3** rewrites the text into a respectful version
4. View the full analysis in the output panels

### Evals Tab

1. Switch to the **Evals** tab
2. Choose a filter: **Quick Run** (10 cases), **All Cases** (50), **Safe Only** (16), or **Unsafe Only** (34)
3. Click **Run Evaluations** — the pipeline processes every test case from `evals/dataset.json`
4. View results:
   - **Overall score** — weighted percentage across all metrics
   - **Score cards** — per-metric pass/total (verdict, category, severity, JSON, rewrite)
   - **Per-case table** — each case with PASS/FAIL badges and time taken
   - **Per-Agent Reports** — accuracy and failed case details for each agent
   - **Deep Analysis** — per-agent tabs (Detector, Analyzer, Rewriter) with failure diagnosis and copy-ready fix suggestions
5. Click **Rerun Failures** to re-test only failed cases
6. Click **Download Report** to save results as a `.txt` file

### Dataset Tab

1. Switch to the **Dataset** tab to view all 50 eval test cases
2. **Search** across all fields in real time
3. Click **Edit** on any case — modify text, verdict, category, severity range, scenario, notes — then click **Save**
4. Click **Add Case** to create a new test case
5. Click **Delete** to remove a case (with confirmation)
6. All changes save directly to `evals/dataset.json`

---

## Evaluations

The project includes an evaluation framework (`evals/runner.py`) that scores the full pipeline against a labeled dataset of **50 test cases** (16 safe texts + 34 hate speech across 10 scenarios and multiple categories).

### What's Measured

| Metric | Weight | Description |
|--------|--------|-------------|
| **Verdict accuracy** | 30% | Did the Detector correctly return `safe` or `unsafe`? |
| **Category match** | 25% | Did the Analyzer identify the correct hate-speech category? |
| **Severity in range** | 15% | Is the severity score within the expected min–max bounds? |
| **JSON validity** | 15% | Did the Analyzer return populated JSON fields (not fallback defaults)? |
| **Rewrite quality** | 15% | Did the Rewriter produce a non-empty output that differs from the original? |

### Running Evals from CLI

```bash
# Run all 50 cases
python -m evals.runner

# Quick run — 10 representative cases
python -m evals.runner --quick

# Run only unsafe cases
python -m evals.runner --unsafe-only

# Run only safe cases
python -m evals.runner --safe-only

# Run specific case IDs
python -m evals.runner --ids 6 7 8

# Save results to JSON
python -m evals.runner --save eval_results.json
```

### Test Dataset: 50 Cases, 10 Scenarios

The labeled dataset (`evals/dataset.json`) covers:

| Scenario | Count | What It Tests |
|----------|-------|---------------|
| `baseline-safe` | 6 | Clearly safe: neutral facts, civic opinions, cultural appreciation |
| `borderline-safe` | 10 | Safe text that mentions sensitive topics (crime stats, refugees, disability) |
| `explicit-hate` | 14 | Obvious hate: slurs, dehumanization, sweeping negative generalizations |
| `coded-language` | 6 | Hate disguised as neutral ("I'm not racist but...", segregation as preference) |
| `microaggression` | 3 | Backhanded compliments, questioning belonging, professional gatekeeping |
| `pseudoscience` | 3 | Racism/sexism framed as biology or "studies show" |
| `sarcastic-hate` | 1 | Irony or sarcasm masking discriminatory intent |
| `casual-hate` | 2 | "Lol typical...", "no offense but..." followed by stereotyping |
| `incitement` | 3 | Eliminationist rhetoric, veiled threats, calls for mass action |
| `multi-target` | 2 | Intersectional hate combining multiple categories in one text |

**Categories covered:** racism (12), xenophobia (7), sexism (6), classism (2), homophobia (2), pseudoscience (3), religious hatred (1), ableism (1), ageism (1), transphobia (1), body shaming (1)

### Per-Agent Deep Analysis

After running evals, the **Deep Analysis** panel provides a structured diagnosis for each agent:

**Agent 1 — Detector:**
- Accuracy by scenario (which scenarios cause false positives / false negatives)
- Specific detection rules to add to the task prompt

**Agent 2 — Analyzer:**
- Category confusion matrix (what the expected category was vs. what was predicted)
- Severity drift analysis (over-rating vs. under-rating)
- JSON quality issues (empty fields, fallback defaults)
- Category disambiguation rules and severity calibration text

**Agent 3 — Rewriter:**
- Empty/short rewrites, identical-to-original failures
- Failures grouped by scenario
- Scenario-specific rewriting guidance

Each section includes **suggested text** for both `agents.py` (backstory additions) and `tasks.py` (task prompt additions) — ready to copy into the source files.

### Why Evals Matter

Evals turn subjective "it seems to work" into **measurable scores**. Here's how they help:

- **Validate severity calibration** — Is the model rating "poor people are useless" as severity 4 or 7? Evals check if scores fall within expected ranges.
- **Catch false negatives** — Sarcastic hate like "Oh sure, THOSE people are so civilized" has no slurs. Evals reveal if the Detector misses it.
- **Verify JSON reliability** — After switching from regex to strict JSON, evals confirm all cases parse cleanly (no broken output).
- **Compare model swaps** — Run evals with Model A, swap to Model B in `agents.py`, run evals again. Two scores, objective comparison.
- **Detect prompt regressions** — Every prompt edit in `tasks.py` can be scored before and after to confirm accuracy improved.
- **Quantify the system** — Instead of guessing, you get: "Verdict: 48/50, Category: 30/34, Overall: 91.2%" — a concrete quality baseline.

---

## Steps We Followed

Below is the step-by-step process we followed to build this project from scratch:

### Step 1: Define the Problem & Agent Roles
We broke the content moderation task into three distinct responsibilities — detection, analysis, and rewriting — and assigned each to a dedicated AI agent with a specific role, goal, and backstory.

### Step 2: Set Up OpenRouter
We created an account on [OpenRouter](https://openrouter.ai), generated an API key, and selected a free model well-suited to all three tasks. The key was stored in a `.env` file.

### Step 3: Create the Agents (`agents.py`)
Using CrewAI's `Agent` class, we defined three agents backed by the same LLM via the `LLM` wrapper with the `openrouter/` prefix. Each agent has a distinct `role`, `goal`, and `backstory` that guides its behavior. Token limits differ per agent: Detector (128), Analyzer (512), Rewriter (256).

### Step 4: Design the Tasks (`tasks.py`)
We wrote detailed task prompts for each agent using CrewAI's `Task` class:
- The **detection task** includes 11 specific categories to check for (slurs, stereotyping, coded language, sarcasm, pseudoscience, incitement, etc.)
- The **analysis task** enforces strict JSON output with keys like `category`, `severity`, `toxic_words`, plus a severity rubric from 1–10
- The **rewrite task** also returns JSON (`{"rewritten_text": "..."}`) with clear rules for preserving intent while removing hatred

### Step 5: Build the Pipeline (`app.py`)
We created a sequential pipeline in Flask: the `/analyze` endpoint receives user text, runs it through the three CrewAI crews in order (detect → analyze → rewrite), parses the strict JSON output from each agent with `json.loads` (with a fallback extractor for markdown-fenced responses), and returns unified JSON to the frontend. We added retry logic with exponential backoff for rate-limited API calls.

### Step 6: Build the Web UI
We embedded a single-page HTML/CSS/JS frontend directly in the Flask app. The UI features a glassmorphism dark theme, animated progress steps that show which agent is currently working, color-coded severity bars, toxic word pills, and a card layout for the respectful rewrite.

### Step 7: Build the Eval Framework
We created 50 labeled test cases across 10 hate speech scenarios. Built a scoring engine that measures 5 weighted metrics (verdict, category, severity, JSON validity, rewrite quality). Added disk caching, Quick Run mode, Rerun Failures, and a live progress bar in the web UI.

### Step 8: Add Per-Agent Deep Analysis
Built a deep analysis engine that diagnoses each agent's failures separately — the Detector's false positives/negatives, the Analyzer's category confusion and severity drift, the Rewriter's empty/short outputs. Each diagnosis includes copy-ready fix suggestions for both `agents.py` and `tasks.py`.

### Step 9: Add Dataset Management & Download Report
Added a third tab (Dataset) to the web UI with full CRUD operations — search, inline edit, add, delete. Added a Download Report button that exports the full eval results and per-agent analysis as a `.txt` file. All dataset changes save directly to `evals/dataset.json`.

---

## Key Learnings

Practical lessons learned while building this project end-to-end:

### 1. CrewAI Agent Overhead Is Real
Each CrewAI agent makes multiple internal LLM calls (delegation checks, tool-use reasoning, retries) beyond the one call you'd expect. A 3-agent pipeline can consume 15–25 API requests per analysis. **Fix:** set `allow_delegation=False` and `max_iter=2` to force minimal execution — cuts calls down to ~3 per analysis with no quality loss.

### 2. Free-Tier Rate Limits Hit Fast
OpenRouter's free tier has request limits. Running evals (50 cases × 3 agents) can exhaust the quota quickly. **Fix:** add disk caching (repeated inputs = 0 calls), a 2-second delay between eval cases to avoid burst 429s, and a Quick Run mode (10 representative cases).

### 3. Model Choice Matters More Than Prompt Length
Choosing the right model matters more than prompt length. **Lesson:** benchmark models with evals before writing longer prompts — the right model solves problems that prompt engineering can't.

### 4. Strict JSON Output > Regex Parsing
Early versions used regex to extract categories and severity from free-text agent responses. This broke on every unexpected format. **Fix:** strict JSON prompts ("Respond with ONLY a valid JSON object") plus a `json.loads` parser with fallback extraction made output 100% reliable.

### 5. max_tokens Prevents Runaway Responses
Without `max_tokens`, models sometimes generate long preambles, disclaimers, or repeated content. Setting caps (Detector: 128, Analyzer: 512, Rewriter: 256) keeps responses focused and reduces token usage without limiting useful output.

### 6. Evals Must Be Fast or They Won't Get Used
The initial eval suite ran all cases sequentially with no caching — it took too long and burned the rate limit. Nobody reruns slow evals. **Fix:** Quick Run mode (10 representative cases), disk caching (skip already-scored cases), and Rerun Failures (only re-test what broke).

### 7. Python Version Compatibility Isn't Obvious
CrewAI doesn't work with Python 3.14 (C extension build failures). This wasn't documented anywhere — we discovered it after install errors. **Lesson:** always pin the Python version in the README (`Requires Python 3.11 or 3.12`) and pin dependency versions in `requirements.txt`.

### 8. Error Messages Should Never Leak Internals
The original `/analyze` endpoint returned raw exception strings (`str(e)`) to the browser. This leaks library paths, API keys in error URLs, and internal stack traces. **Fix:** log the full traceback server-side, return a generic "Analysis failed. Please try again." to the client.

### 9. Separate What Changes from What Doesn't
Moving HTML from inline Python strings to `templates/index.html`, eval files to `evals/`, and config to `.env.example` made the project dramatically easier to navigate. When someone opens the repo, they can instantly tell: "3 Python files = core app, evals/ = testing, templates/ = UI."

### 10. Cache Eval Results to Disk
Running 50 eval cases through 3 agents = many API calls. The eval disk cache (`evals/cache.json`) stores already-scored results so reruns skip completed cases. This makes iterative debugging practical: fix a prompt, rerun only the failures, keep the passing results cached.

### 11. Per-Agent Analysis Changes Everything
A single overall score ("87%") doesn't tell you which agent to fix. Breaking failures down by agent — with specific scenario breakdowns, category confusion, and severity drift — makes improvement actionable. The deep analysis system generates the exact text to add to each agent's backstory or task prompt.

---

## Example Inputs

| Text | Expected Result |
|------|-----------------|
| "Public libraries are important resources for everyone." | ✅ Safe |
| "People from that country are all criminals and should be deported." | ❌ Xenophobia (severity 6–10) |
| "That race is inherently less intelligent — it's basic biology." | ❌ Racism / Pseudoscience (severity 7–9) |
| "I'm not racist but those people just don't fit in here." | ❌ Coded Language (severity 4–6) |
| "You're so articulate for someone from that background." | ❌ Microaggression (severity 2–4) |

---

## Screenshots

<!-- Add screenshots of the Analyze tab, Evals tab, and Dataset tab here -->

---

## License

MIT — see [LICENSE](LICENSE) for details.