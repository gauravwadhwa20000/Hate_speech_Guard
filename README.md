<div align="center">

# 🛡️ Hate Speech Guardian

**A multi-agent AI pipeline that detects, analyzes, and rewrites hate speech — built with [CrewAI](https://crewai.com/) and powered by free LLMs through [OpenRouter](https://openrouter.ai/).**

[![Demo Video](https://img.shields.io/badge/▶_Watch_Demo-YouTube-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/watch?v=xNqT8rf-5Jc)
[![GitHub Repo](https://img.shields.io/badge/Source_Code-GitHub-black?style=for-the-badge&logo=github)](https://github.com/gauravwadhwa20000/Hate_speech_Guard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=for-the-badge)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-yellow?style=for-the-badge&logo=python)](https://www.python.org/)

</div>

---

## 📺 Video Demo

Want to see the project in action before cloning? Watch the full walkthrough:

[![Hate Speech Guardian Demo](https://img.youtube.com/vi/xNqT8rf-5Jc/maxresdefault.jpg)](https://www.youtube.com/watch?v=xNqT8rf-5Jc)

> **[▶ Watch on YouTube](https://www.youtube.com/watch?v=xNqT8rf-5Jc)** — covers the architecture, live demo, evaluation framework, and how each agent works.

---

## 💡 What Is This?

Online platforms struggle with content moderation at scale. Manual review is slow, expensive, and inconsistent. A single LLM prompt can miss nuance — sarcasm, coded language, and microaggressions often slip through.

**Hate Speech Guardian** takes a different approach. Instead of relying on one model for everything, it breaks the problem into three focused stages, each handled by a specialized AI agent:

| Stage | Agent | What It Does |
|-------|-------|-------------|
| **1. Detection** | 🛡️ Hate Speech Detector | Reads the text and makes a binary call — **safe** or **unsafe** |
| **2. Analysis** | 📊 Content Analyzer | If unsafe, classifies the category, rates severity (1–10), identifies toxic words, and explains why |
| **3. Rewriting** | ✏️ Content Rewriter | Transforms the hateful text into a respectful alternative while keeping the speaker's underlying point |

The three agents are orchestrated through a **[CrewAI](https://crewai.com/) sequential pipeline** and the results are displayed in a clean Flask web UI. On top of that, there's a full **evaluation framework** with 50 labeled test cases so you can measure exactly how well each agent performs — and know what to fix.

---

## 🏗️ Architecture

<div align="center">

```
User enters text in the Web UI
              │
              ▼
┌─────────────────────────────┐
│   Agent 1 — Detector        │  Classifies input as "safe" or "unsafe"
│   Role: Content Safety      │
└──────────────┬──────────────┘
               │ unsafe?
               ▼
┌─────────────────────────────┐
│   Agent 2 — Analyzer        │  Category, severity, intent, toxic words
│   Role: Sociolinguistics    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│   Agent 3 — Rewriter        │  Rewrites text into a respectful version
│   Role: Content Moderator   │
└──────────────┬──────────────┘
               │
               ▼
        JSON response → Web UI
```

</div>

If the Detector classifies the text as **safe**, the pipeline **short-circuits** — no analysis or rewrite happens, saving time and API calls.

---

## 📸 Screenshot

<div align="center">

<img src="assets/3-agents-pipeline.png" alt="Hate Speech Guardian — 3 Agent Pipeline (Detection → Analysis → Rewriting)" width="800">

</div>

> The three AI agents working in sequence — **Agent 1** decides if the text is safe or unsafe, **Agent 2** returns category, severity, intent, and toxic words as structured JSON, and **Agent 3** converts harmful text into a respectful alternative. Built with [CrewAI](https://crewai.com/) and [OpenRouter](https://openrouter.ai/).
>
> **See the full app in action?** [▶ Watch the demo video on YouTube](https://www.youtube.com/watch?v=xNqT8rf-5Jc)

---

## 🚀 Quick Start

> **Requires Python 3.11 or 3.12.** CrewAI is not compatible with Python 3.14. Check with `python --version`.

### 1. Clone the repository

```bash
git clone https://github.com/gauravwadhwa20000/Hate_speech_Guard.git
cd Hate_speech_Guard
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Linux / Mac
source venv/bin/activate

# Windows PowerShell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Get your free OpenRouter API key

1. Go to **[openrouter.ai/keys](https://openrouter.ai/keys)** (free, no credit card needed)
2. Create an API key
3. Create a `.env` file in the project root:

```bash
cp .env.example .env            # Linux/Mac
copy .env.example .env          # Windows
```

4. Paste your key:

```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### 5. Start the app

```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser — you're ready to go!

---

## 🎯 How to Use

The web UI has **three tabs**:

### 🛡️ Analyze Tab

1. Type or paste any text into the input box
2. Click **Analyze Text**
3. Watch the three agents process your text one by one:
   - **Agent 1** decides if the text is safe or contains hate speech
   - **Agent 2** digs deeper — classifies the category, rates severity, and highlights the toxic words
   - **Agent 3** rewrites the text into something respectful
4. View the full breakdown: verdict, category, severity bar, toxic word pills, explanation, and the rewritten version

### 📊 Evals Tab

1. Switch to the **Evals** tab
2. Pick a filter — **Quick Run** (10 cases), **All** (50), **Safe Only**, or **Unsafe Only**
3. Hit **Run Evaluations** — the pipeline processes every test case
4. Review the results:
   - **Overall score** — weighted percentage across all five metrics
   - **Score cards** — pass/total for verdict, category, severity, JSON validity, rewrite quality
   - **Per-case table** — every test case with PASS/FAIL badges
   - **Per-Agent Reports** — accuracy and specific failures for each agent
   - **Deep Analysis** — failure diagnosis with copy-ready fix suggestions
5. Use **Rerun Failures** to re-test only the broken cases
6. Hit **Download Report** to save everything as a `.txt` file

### 📝 Dataset Tab

1. Browse all 50 eval test cases
2. **Search** across all fields in real time
3. **Edit** any case inline — text, verdict, category, severity range, scenario, notes
4. **Add** new test cases or **Delete** existing ones
5. All changes save directly to `evals/dataset.json`

---

## 🤖 The Three Agents

Each agent has a specific role, goal, and backstory that shapes how it thinks. They're defined in `agents.py` using [CrewAI](https://crewai.com/)'s `Agent` class.

### Agent 1 — Hate Speech Detector

> *"You are a content safety specialist trained to identify hate speech, offensive language, threats, and discriminatory content."*

The Detector checks text against **11 specific rules** and makes a binary call: **safe** or **unsafe**.

| # | Rule | What It Catches |
|---|------|-----------------|
| 1 | Explicit Hate | Slurs, racial epithets, derogatory terms |
| 2 | Generalizations | "All X are Y", "X people always..." |
| 3 | Dehumanization | Comparing groups to animals, diseases, vermin |
| 4 | Stereotyping | Portraying a group as criminal, inferior, dangerous |
| 5 | Threats & Incitement | Calls for violence, exclusion, or harm |
| 6 | Coded Language | "Go back to your country", dog whistles |
| 7 | Sarcastic Hate | Irony masking hateful intent |
| 8 | Implied Discrimination | Suggesting a group doesn't belong |
| 9 | Class-Based Hate | Targeting based on economic status |
| 10 | Ageism / Ableism | Mocking based on age or disability |
| 11 | Pseudoscientific Racism | "Racial IQ differences are science" |

### Agent 2 — Content Analyzer

> *"You are a sociolinguistics expert who specializes in content analysis."*

When text is flagged as unsafe, the Analyzer returns a structured JSON object with:

- **Category** — racism, sexism, xenophobia, homophobia, ableism, classism, etc.
- **Severity** — 1 (microaggression) to 10 (threats of violence)
- **Intent** — deliberate, ignorant, casual, or satirical
- **Confidence** — how sure the model is (1–10)
- **Toxic words** — the exact words/phrases that make the text harmful
- **Explanation** — why the text is classified this way

### Agent 3 — Content Rewriter

> *"You are a skilled content moderator and empathetic communicator."*

The Rewriter takes hateful text and transforms it into a respectful, constructive alternative. The key rule: **preserve the speaker's underlying concern** while removing the hatred. "Those immigrants are ruining our jobs" becomes something like "I'm concerned about the impact of immigration on local employment opportunities."

---

## 🔧 Tech Stack

| Component | Technology |
|-----------|------------|
| AI Framework | [CrewAI](https://crewai.com/) — multi-agent orchestration |
| LLM Routing | [LiteLLM](https://github.com/BerriAI/litellm) — enables CrewAI → OpenRouter |
| LLM Gateway | [OpenRouter](https://openrouter.ai/) — unified access to 200+ models |
| Model | `arcee-ai/trinity-large-preview:free` (all 3 agents, $0 cost) |
| Backend | [Flask](https://flask.palletsprojects.com/) |
| Language | Python 3.11+ |
| Frontend | Single-page HTML/CSS/JS with glassmorphism dark theme |

---

## 🌐 Why OpenRouter?

[**OpenRouter**](https://openrouter.ai/) is a unified API gateway that gives you access to **200+ LLMs** from OpenAI, Meta, Google, NVIDIA, Mistral, and more — all through a single API key.

We chose it for this project because:

- **Free models available** — many models have a `:free` suffix, so you can prototype without spending anything
- **Single API key** — no need to manage accounts with multiple providers
- **Easy model switching** — change one string in `agents.py` and you're using a different model
- **CrewAI integration** — [CrewAI](https://crewai.com/)'s `LLM` class supports OpenRouter natively with the `openrouter/` prefix

Here's how each agent connects to OpenRouter in `agents.py`:

```python
from crewai.llm import LLM

detector_llm = LLM(
    model="openrouter/arcee-ai/trinity-large-preview:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    max_tokens=128,
)
```

The `openrouter/` prefix tells CrewAI (via LiteLLM) to route requests through OpenRouter's API. The API key lives in a `.env` file — never hardcoded.

---

## 📁 Project Structure

```
Hate_speech_Guard/
├── app.py                  # Flask server + 3-agent pipeline + eval engine + dataset API
├── agents.py               # Agent definitions (Detector, Analyzer, Rewriter)
├── tasks.py                # Task prompts with detection rules, JSON schemas, severity rubric
├── evals/
│   ├── runner.py           # CLI evaluation runner
│   ├── dataset.json        # 50 labeled test cases (16 safe + 34 unsafe)
│   └── cache.json          # Disk cache for eval results (auto-generated)
├── templates/
│   └── index.html          # Web UI (Analyze + Evals + Dataset tabs)
├── .env.example            # Environment variable template
├── requirements.txt        # Python dependencies
├── test_models.py          # Model connectivity test
├── LICENSE                 # MIT License
└── README.md               # You're reading this
```

---

## 📏 Severity Scale

When the Analyzer assigns a severity score, it follows this calibration:

| Score | Level | What It Means | Example |
|-------|-------|---------------|---------|
| **1–2** | Mild | Microaggressions, thoughtless remarks | "You're so articulate for a..." |
| **3–4** | Moderate | Clear bias, casual stereotyping | "Women just aren't as logical" |
| **5–6** | High | Explicit slurs, derogatory generalizations | "All X are criminals" |
| **7–8** | Severe | Dehumanization, incitement to discrimination | Comparing people to animals |
| **9–10** | Extreme | Direct threats of violence, calls for genocide | "They should all be..." |

Each eval test case has expected min/max severity bounds. The eval system flags scores that are **too high** (over-rating) or **too low** (under-rating).

---

## 📊 Evaluation Framework

The project includes a built-in eval system with **50 labeled test cases** across 10 hate speech scenarios. This is what turns "it seems to work" into actual, measurable numbers.

### Five Weighted Metrics

| Metric | Weight | What It Checks |
|--------|--------|----------------|
| **Verdict accuracy** | 30% | Did the Detector correctly return safe or unsafe? |
| **Category match** | 25% | Did the Analyzer identify the right category? |
| **Severity in range** | 15% | Is the severity within expected bounds? |
| **JSON validity** | 15% | Did the Analyzer return proper structured output? |
| **Rewrite quality** | 15% | Did the Rewriter produce a meaningful alternative? |

### Test Scenarios

| Scenario | Cases | What It Tests |
|----------|-------|---------------|
| `baseline-safe` | 6 | Clearly safe: neutral facts, civic opinions |
| `borderline-safe` | 10 | Safe text mentioning sensitive topics |
| `explicit-hate` | 14 | Obvious hate: slurs, dehumanization |
| `coded-language` | 6 | Hate disguised as neutral speech |
| `microaggression` | 3 | Backhanded compliments, gatekeeping |
| `pseudoscience` | 3 | Racism framed as "biology" or "science" |
| `sarcastic-hate` | 1 | Irony masking discriminatory intent |
| `casual-hate` | 2 | "No offense but..." followed by stereotyping |
| `incitement` | 3 | Veiled threats, calls for mass action |
| `multi-target` | 2 | Intersectional hate across multiple categories |

### Running Evals from the Command Line

```bash
python -m evals.runner              # All 50 cases
python -m evals.runner --quick      # 10 representative cases
python -m evals.runner --unsafe-only
python -m evals.runner --safe-only
python -m evals.runner --ids 6 7 8  # Specific case IDs
python -m evals.runner --save results.json
```

### Deep Analysis

After running evals, the Deep Analysis panel diagnoses each agent individually:

- **Detector** — false positive/negative breakdown by scenario, detection rules to add
- **Analyzer** — category confusion, severity drift, JSON quality issues, calibration fixes
- **Rewriter** — empty/short rewrites, failures by scenario, rewriting guidance

Each diagnosis includes **copy-ready suggested text** for both `agents.py` and `tasks.py` — paste it in to improve your scores.

---

## 🧪 Try It Yourself

| Input Text | Expected Result |
|------------|-----------------|
| "Public libraries are important resources for everyone." | ✅ Safe |
| "People from that country are all criminals and should be deported." | ❌ Xenophobia (severity 6–10) |
| "That race is inherently less intelligent — it's basic biology." | ❌ Racism / Pseudoscience (severity 7–9) |
| "I'm not racist but those people just don't fit in here." | ❌ Coded Language (severity 4–6) |
| "You're so articulate for someone from that background." | ❌ Microaggression (severity 2–4) |

---

## 🔨 How We Built It — Step by Step

Here's the process we followed to build this project from scratch:

**Step 1 — Define the problem.** We broke content moderation into three responsibilities (detect, analyze, rewrite) and assigned each to a dedicated AI agent.

**Step 2 — Set up OpenRouter.** Created a free account on [openrouter.ai](https://openrouter.ai/), grabbed an API key, and picked a free model that handles all three tasks well.

**Step 3 — Build the agents (`agents.py`).** Used [CrewAI](https://crewai.com/)'s `Agent` class to define three agents, each with a unique role, goal, and backstory. Connected them to OpenRouter via the `LLM` wrapper.

**Step 4 — Design the tasks (`tasks.py`).** Wrote detailed prompts for each agent: 11 detection rules for the Detector, strict JSON output schema for the Analyzer, and respectful-rewrite guidelines for the Rewriter.

**Step 5 — Wire up the pipeline (`app.py`).** Built a Flask endpoint that runs the three agents sequentially, parses their JSON output (with fallback for messy LLM responses), and returns unified results. Added retry logic with exponential backoff for rate limits.

**Step 6 — Build the web UI.** Created a single-page frontend with a glassmorphism dark theme, animated progress steps, color-coded severity bars, and toxic word highlighting.

**Step 7 — Build the eval framework.** Labeled 50 test cases across 10 scenarios. Built a scoring engine with 5 weighted metrics, disk caching, Quick Run mode, and a live progress bar.

**Step 8 — Add deep analysis.** Built a per-agent diagnosis system that pinpoints each agent's failures and generates copy-ready fix suggestions for the prompts and backstories.

**Step 9 — Add dataset management.** Added a Dataset tab with full CRUD — search, inline editing, add, delete. Plus a Download Report button that exports everything as a `.txt` file.

---

## 💡 Key Learnings

Things we learned the hard way so you don't have to:

| # | Lesson | What We Did |
|---|--------|-------------|
| 1 | **CrewAI agent overhead is real** | Each agent makes multiple internal LLM calls. Set `allow_delegation=False` and `max_iter=2` to cut calls from 15–25 down to ~3. |
| 2 | **Free-tier rate limits hit fast** | Added disk caching, 2s delay between eval cases, and Quick Run mode (10 cases). |
| 3 | **Model choice > prompt length** | Benchmark models with evals first — the right model solves problems that longer prompts can't. |
| 4 | **Strict JSON > regex parsing** | "Respond with ONLY a valid JSON object" + `json.loads` with fallback extraction = 100% reliable. |
| 5 | **Set max_tokens always** | Without it, models generate long preambles. Caps: Detector 128, Analyzer 512, Rewriter 256. |
| 6 | **Evals must be fast** | Nobody reruns slow evals. Quick Run + caching + Rerun Failures made iteration practical. |
| 7 | **Pin your Python version** | CrewAI breaks on Python 3.14. We discovered this after install errors. Use 3.11 or 3.12. |
| 8 | **Never leak error internals** | Return generic messages to the client. Log full tracebacks server-side only. |
| 9 | **Organize files early** | `app.py` + `agents.py` + `tasks.py` + `evals/` + `templates/` — someone new can navigate instantly. |
| 10 | **Cache eval results** | Disk cache means reruns skip already-scored cases. Fix a prompt, rerun only failures. |
| 11 | **Per-agent analysis changes everything** | An overall score doesn't tell you which agent to fix. Breaking it down by agent makes improvement actionable. |

---

## 📜 License

MIT — see [LICENSE](LICENSE) for details.

---

<div align="center">

**Built with [CrewAI](https://crewai.com/) and [OpenRouter](https://openrouter.ai/)**

[⭐ Star this repo](https://github.com/gauravwadhwa20000/Hate_speech_Guard/) if you found it useful!

</div>
