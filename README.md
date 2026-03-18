# Hate Speech Guardian

A multi-agent AI content moderation pipeline built with **CrewAI** and **Flask**. Three specialized AI agents collaborate in sequence to **detect**, **analyze**, and **rewrite** hate speech in user-submitted text — all powered by free LLMs accessed through **OpenRouter**.

The goal of this project is to demonstrate how multiple AI agents, each with a distinct role and backed by a different language model, can be orchestrated into a cooperative pipeline to tackle a real-world problem: online hate speech moderation.

---

## Project Overview

Online platforms struggle with content moderation at scale. Manual review is slow and expensive; a single LLM prompt can miss nuance. **Hate Speech Guardian** solves this by breaking the moderation task into three focused stages, each handled by a dedicated AI agent:

1. **Detection** — A safety-focused model decides whether the text is *safe* or *unsafe*.
2. **Analysis** — If unsafe, a reasoning model returns a strict JSON object with the hate category, severity (1–10), intent, target group, confidence score, and the exact toxic words.
3. **Rewriting** — A creative model returns a JSON object containing the hateful text rewritten as a respectful, constructive alternative while preserving the speaker's underlying concern.

The agents communicate through a **CrewAI** sequential pipeline, and results are displayed in a modern **Flask** web UI with animated progress indicators.

---

## How It Works

```
User enters text in the Web UI
           │
           ▼
┌──────────────────────────┐
│  Agent 1 — Detector      │  nvidia/nemotron-nano-9b-v2 (free)
│  Role: Content Safety     │  Classifies input as "safe" or "unsafe"
└────────────┬─────────────┘
             │ unsafe?
             ▼
┌──────────────────────────┐
│  Agent 2 — Analyzer      │  z-ai/glm-4.5-air (free)
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

If the Detector classifies the text as **safe**, the pipeline short-circuits and returns immediately — no analysis or rewrite is performed.

---

## Models

Each agent uses a **different free model** from a different provider, all accessed via **OpenRouter**. This showcases how you can mix-and-match models based on their strengths:

| Agent | Model | Provider | Why This Model? |
|-------|-------|----------|-----------------|
| **Detector** | `nvidia/nemotron-nano-9b-v2:free` | NVIDIA | Fast and lightweight; ideal for binary safe/unsafe classification |
| **Analyzer** | `z-ai/glm-4.5-air:free` | Zhipu AI | Strong reasoning and structured output; good at following exact field formats |
| **Rewriter** | `arcee-ai/trinity-large-preview:free` | Arcee AI | Fast inference and strong instruction-following; produces clean, respectful rewrites |

All models are hosted on [OpenRouter](https://openrouter.ai) and accessed through their unified API. The `:free` suffix means they cost **$0** — no billing required.

### Why Different Models?

Using three different models is a deliberate design choice:
- **Specialization** — Each model handles what it's best at (classification vs. analysis vs. creative writing).
- **Resilience** — If one provider has downtime, only that stage is affected.
- **Cost optimization** — Free-tier models keep the entire pipeline at zero cost.

---

## OpenRouter

[OpenRouter](https://openrouter.ai) is a unified API gateway that gives access to 200+ LLMs from providers like OpenAI, Meta, Google, NVIDIA, Mistral, and many more — all through a single API key and a consistent API format (compatible with the OpenAI SDK).

### Why OpenRouter?

- **Single API key** — Access models from dozens of providers without managing multiple accounts.
- **Free models** — Many models are available at no cost (suffixed with `:free`), perfect for prototyping and demos.
- **Easy switching** — Change the model by editing a single string (e.g., swap `nvidia/nemotron-nano-9b-v2:free` for `arcee-ai/trinity-large-preview:free`).
- **CrewAI integration** — CrewAI's `LLM` class supports OpenRouter out of the box using the `openrouter/` prefix.

### How It's Used in This Project

In [agents.py](agents.py), each agent is initialized with a different OpenRouter model:

```python
from crewai.llm import LLM

detector_llm = LLM(
    model="openrouter/nvidia/nemotron-nano-9b-v2:free",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)
```

The `openrouter/` prefix tells CrewAI (via **LiteLLM**) to route the request through the OpenRouter API. CrewAI uses LiteLLM under the hood for non-native providers — it must be installed separately (`pip install litellm`). The API key is stored in a `.env` file and loaded via `python-dotenv`.

---

## Features

- **3-Agent Pipeline** — Demonstrates CrewAI multi-agent orchestration with sequential task execution
- **Hate Speech Detection** — Binary safe/unsafe classification with detailed detection criteria (slurs, stereotyping, coded language, sarcasm, etc.)
- **Category Classification** — Identifies type: racism, sexism, xenophobia, religious hatred, homophobia, ableism, classism, and more
- **Severity Rating** — Visual severity bar from 1 (mild microaggression) to 10 (threats of violence)
- **Intent Classification** — Distinguishes deliberate hatred from ignorant, casual, or satirical speech
- **Confidence Score** — The analyzer reports how confident it is in its own assessment (1–10)
- **Toxic Word Highlighting** — Pinpoints the exact offensive words/phrases from the input
- **Content Rewriting** — Transforms hateful text into respectful alternatives while preserving the speaker's underlying concern
- **Retry with Backoff** — Automatic retry (up to 5 attempts) with exponential backoff for rate-limited API calls (HTTP 429)
- **Modern Web UI** — Clean dark glassmorphism theme with animated agent progress steps

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| AI Framework | [CrewAI](https://github.com/crewAIInc/crewAI) (multi-agent orchestration) |
| LLM Routing | [LiteLLM](https://github.com/BerriAI/litellm) (enables CrewAI → OpenRouter) |
| LLM Gateway | [OpenRouter](https://openrouter.ai) (unified access to 200+ models) |
| Detection Model | `nvidia/nemotron-nano-9b-v2:free` |
| Analysis Model | `z-ai/glm-4.5-air:free` |
| Rewrite Model | `arcee-ai/trinity-large-preview:free` |
| Web Framework | Flask |
| Language | Python 3.11+ |

---

## Steps We Followed

Below is the step-by-step process we followed to build this project from scratch:

### Step 1: Define the Problem & Agent Roles
We broke the content moderation task into three distinct responsibilities — detection, analysis, and rewriting — and assigned each to a dedicated AI agent with a specific role, goal, and backstory.

### Step 2: Set Up OpenRouter
We created an account on [OpenRouter](https://openrouter.ai), generated an API key, and selected three free models well-suited to each agent's task. The key was stored in a `.env` file.

### Step 3: Create the Agents (`agents.py`)
Using CrewAI's `Agent` class, we defined three agents, each backed by a different LLM via the `LLM` wrapper with the `openrouter/` prefix. Each agent has a distinct `role`, `goal`, and `backstory` that guides its behavior.

### Step 4: Design the Tasks (`tasks.py`)
We wrote detailed task prompts for each agent using CrewAI's `Task` class. The detection task includes 10 specific categories to check for (slurs, stereotyping, coded language, sarcasm, etc.). The analysis task enforces **strict JSON output** with keys like `category`, `severity`, `toxic_words`. The rewrite task also returns JSON (`{"rewritten_text": "..."}`) with clear rules for preserving intent while removing hatred.

### Step 5: Build the Pipeline (`app.py`)
We created a sequential pipeline in Flask: the `/analyze` endpoint receives user text, runs it through the three CrewAI crews in order (detect → analyze → rewrite), parses the strict JSON output from each agent with `json.loads` (with a fallback extractor for markdown-fenced responses), and returns unified JSON to the frontend. We added retry logic with exponential backoff for rate-limited API calls.

### Step 6: Build the Web UI
We embedded a single-page HTML/CSS/JS frontend directly in the Flask app. The UI features a glassmorphism dark theme, animated progress steps that show which agent is currently working, color-coded severity bars, toxic word pills, and a card layout for the respectful rewrite.

### Step 7: Test & Iterate
We tested with a variety of inputs — safe text, mildly insensitive text, explicit hate speech, coded language, and sarcastic hate — and refined the prompts to reduce false positives and improve structured output parsing.

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

## Usage

The UI has two tabs:

### Analyze Tab
1. Enter any text in the input box
2. Click **Analyze Text**
3. Watch the 3 agents process sequentially:
   - **Agent 1** detects if the text is safe or contains hate speech
   - **Agent 2** classifies the category, rates severity, and highlights toxic words
   - **Agent 3** rewrites the text into a respectful version
4. View the full analysis in the output panels

### Analytics Tab
1. Switch to the **Analytics** tab
2. Choose a filter: All Cases, Safe Only, or Unsafe Only
3. Click **Run Evaluations** — the pipeline processes every test case from `evals/dataset.json`
4. View results:
   - **Overall score** — weighted percentage across all metrics
   - **Score cards** — per-metric pass/total (verdict, category, severity, JSON, rewrite)
   - **Per-case table** — shows each case with PASS/FAIL badges and time taken
5. A progress bar with live status updates shows which case is being evaluated

### Example Inputs

| Text | Expected Result |
|------|----------------|
| "Public libraries are important resources for everyone." | ✅ Safe |
| "That country is less intelligent and incapable of contributing to society." | ❌ Hate Speech — Xenophobia |
| "People from that race are all untrustworthy and should not be allowed to work." | ❌ Hate Speech — Racism |

## Project Structure

```
hate-speech-guardian/
├── app.py                  # Flask web server + pipeline orchestration
├── agents.py               # 3 CrewAI agents (Detector, Analyzer, Rewriter)
├── tasks.py                # Task definitions for each agent
├── evals/
│   ├── runner.py           # Evaluation runner — scores pipeline against test cases
│   └── dataset.json        # 20 labeled test cases (5 safe + 15 unsafe)
├── templates/
│   └── index.html          # Web UI (Analyze + Analytics tabs)
├── .env.example            # Environment variable template — copy to .env
├── requirements.txt        # Pinned Python dependencies
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # This file
```

---

## Evaluations

The project includes an evaluation framework (`evals/runner.py`) that scores the full pipeline against a labeled dataset of 20 test cases (5 safe texts + 15 hate speech across multiple categories).

### What's Measured

| Metric | Weight | Description |
|--------|--------|-------------|
| **Verdict accuracy** | 30% | Did the Detector correctly return `safe` or `unsafe`? |
| **Category match** | 25% | Did the Analyzer identify the correct hate-speech category? |
| **Severity in range** | 15% | Is the severity score within the expected min–max bounds? |
| **JSON validity** | 15% | Did the Analyzer return populated JSON fields (not fallback defaults)? |
| **Rewrite quality** | 15% | Did the Rewriter produce a non-empty output that differs from the original? |

### Running Evals

```bash
# Run all 20 cases
python -m evals.runner

# Quick run — 8 representative cases
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

### Test Dataset

The labeled dataset (`evals/dataset.json`) covers:
- **Safe texts** — neutral statements, constructive opinions, cross-cultural positivity
- **Racism** — generalizations, dehumanization, pseudoscientific claims
- **Sexism** — stereotypes about women in leadership
- **Xenophobia** — anti-immigrant sentiment, coded "go back" language
- **Religious hatred** — sweeping generalizations about religious groups
- **Homophobia / Transphobia** — explicit hate with calls to restrict rights
- **Ageism / Ableism** — devaluing elderly or disabled people
- **Classism** — blaming poverty on character
- **Body shaming** — derogatory comments about body type
- **Sarcastic hate** — irony masking discriminatory intent

### Why Evals Matter

Evals turn subjective "it seems to work" into measurable scores. Here's how they help this project:

- **Validate severity calibration** — Is the model rating "poor people are useless" as severity 4 or 7? Evals check if scores fall within expected ranges.
- **Catch false negatives** — Sarcastic hate like *"Oh sure, THOSE people are so civilized"* has no slurs. Evals reveal if the Detector misses it.
- **Verify JSON reliability** — After switching from regex to strict JSON, evals confirm all 20 cases parse cleanly (no broken output).
- **Compare model swaps** — Run evals with Model A, swap to Model B in `agents.py`, run evals again. Two scores, objective comparison.
- **Detect prompt regressions** — Every prompt edit in `tasks.py` can be scored before and after to confirm accuracy improved.
- **Quantify the system** — Instead of guessing, you get: *"Verdict: 18/20, Category: 13/15, Overall: 87.3%"* — a concrete quality baseline.

---

## Key Learnings

Practical lessons learned while building this project end-to-end:

### 1. CrewAI Agent Overhead Is Real
Each CrewAI agent makes **multiple internal LLM calls** (delegation checks, tool-use reasoning, retries) beyond the one call you'd expect. A 3-agent pipeline can consume 15–25 API requests per analysis. Fix: set `allow_delegation=False` and `max_iter=1` to force one-shot execution — cuts calls down to ~3 per analysis with no quality loss.

### 2. Free-Tier Rate Limits Hit Fast
OpenRouter's free tier allows 50 requests/day. Running evals (20 cases × 3 agents) can exhaust the quota in a single run. Fix: add in-memory caching (repeated inputs = 0 calls), disk caching for evals, and a 2-second delay between eval cases to avoid burst 429s.

### 3. Model Choice Matters More Than Prompt Length
Choosing the right model matters more than prompt length. Lesson: benchmark models with evals before writing longer prompts — the right model solves problems that prompt engineering can't.

### 4. Strict JSON Output > Regex Parsing
Early versions used regex to extract categories and severity from free-text agent responses. This broke on every unexpected format. Switching to **strict JSON prompts** (`"Respond with ONLY a valid JSON object"`) plus a `json.loads` parser with fallback extraction made output 100% reliable.

### 5. `max_tokens` Prevents Runaway Costs
Without `max_tokens`, models sometimes generate long preambles, disclaimers, or repeated content. Setting caps (Detector: 32, Analyzer: 512, Rewriter: 256) keeps responses focused and reduces token usage without limiting useful output.

### 6. Evals Must Be Fast or They Won't Get Used
The initial eval suite ran all 20 cases sequentially with no caching — it took 10+ minutes and burned the daily rate limit. Nobody reruns slow evals. Fix: Quick Run mode (8 representative cases), disk caching (skip already-scored cases), and Rerun Failures (only re-test what broke).

### 7. Python Version Compatibility Isn't Obvious
CrewAI doesn't work with Python 3.14 (C extension build failures). This wasn't documented anywhere — we discovered it after install errors. Lesson: always pin the Python version in the README (`Requires Python 3.11 or 3.12`) and pin dependency versions in `requirements.txt`.

### 8. Error Messages Should Never Leak Internals
The original `/analyze` endpoint returned raw exception strings (`str(e)`) to the browser. This leaks library paths, API keys in error URLs, and internal stack traces. Fix: log the full traceback server-side, return a generic `"Analysis failed. Please try again."` to the client.

### 9. Separate What Changes from What Doesn't
Moving HTML from inline Python strings to `templates/index.html`, eval files to `evals/`, and config to `.env.example` made the project dramatically easier to navigate. When someone opens the repo, they can instantly tell: *"3 Python files = core app, evals/ = testing, templates/ = UI."*

### 10. Cache Eval Results to Disk
Running 20 eval cases through 3 agents = 60 API calls. The eval disk cache (`evals/cache.json`) stores already-scored results so reruns skip completed cases. This makes iterative debugging practical: fix a prompt, rerun only the failures, keep the passing results cached.

## Screenshots

Add screenshots of the app to the `screenshots/` folder and reference them here:

<!-- Uncomment and update the paths after adding screenshots:
![Analyze Tab](screenshots/analyze-tab.png)
![Hate Speech Detected](screenshots/hate-speech-detected.png)
![Analytics Tab](screenshots/analytics-tab.png)
-->

---

## License

MIT — see [LICENSE](LICENSE) for details.
