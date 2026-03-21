# 🚀 Hate Speech Guardian

A **production-style multi-agent AI moderation system** built with **CrewAI + OpenRouter + Flask** that:

✅ Detects hate speech
✅ Analyzes category, severity, intent
✅ Rewrites harmful content into respectful language
✅ Evaluates itself with a structured scoring framework

---

## 🎥 Demo

https://www.youtube.com/watch?v=xNqT8rf-5Jc


## 📌 Why This Project Matters

Most AI moderation systems rely on **a single prompt**, which leads to:

* Inconsistent outputs
* Poor handling of nuance (sarcasm, coded language)
* No measurable performance

This project solves that by building a **modular AI pipeline** with:

* Specialized agents
* Structured outputs (JSON)
* A built-in evaluation system

---

## 🧠 System Architecture

```text
User Input
   │
   ▼
[Agent 1: Detector]
   │
   ├── Safe → Return immediately
   │
   ▼
[Agent 2: Analyzer]
   ▼
[Agent 3: Rewriter]
   ▼
Final Structured JSON Response
```

---

## 🤖 Agents Explained

### 🛡️ Detector Agent

* Classifies input as `safe` or `unsafe`
* Detects:

  * Slurs
  * Generalizations
  * Coded language
  * Sarcasm
  * Threats

---

### 🔍 Analyzer Agent

```json
{
  "category": "racism",
  "severity": 7,
  "intent": "harmful",
  "target_group": "...",
  "confidence": 9,
  "toxic_words": ["..."]
}
```

---

### ✍️ Rewriter Agent

```json
{
  "rewritten_text": "Respectful and constructive version"
}
```

---

## ⚙️ Tech Stack

| Component    | Technology   |
| ------------ | ------------ |
| AI Framework | CrewAI       |
| LLM Gateway  | OpenRouter   |
| Routing      | LiteLLM      |
| Backend      | Flask        |
| Language     | Python 3.11+ |

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/gauravwadhwa20000/Hate_speech_Guard.git
cd Hate_speech_Guard
```

---

### 2. Setup Environment

```bash
python -m venv venv
```

Activate:

```bash
# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Setup OpenRouter

1. Get API key: [https://openrouter.ai/keys](https://openrouter.ai/keys)
2. Create `.env`:

```env
OPENROUTER_API_KEY=your_key_here
```

---

## ▶️ Run Locally

```bash
python app.py
```

Open:

```
http://127.0.0.1:5000
```

---

## 🏗️ How This Was Built

### 1. Problem Breakdown

Split moderation into:

* Detection
* Analysis
* Rewriting

---

### 2. Agent Design

Each agent:

* Single responsibility
* Clear role
* Structured output

---

### 3. Task Engineering

* Strict JSON outputs
* Defined taxonomy + severity rules

---

### 4. Pipeline Execution

* Sequential flow
* Short-circuit for safe content
* JSON parsing + fallback

---

### 5. Evaluation Layer

* Dataset (50 cases)
* Weighted scoring system
* Per-agent analysis

---

## 📊 Evaluation System

### Metrics

| Metric             | Weight |
| ------------------ | ------ |
| Verdict Accuracy   | 30%    |
| Category Accuracy  | 25%    |
| Severity Alignment | 15%    |
| JSON Validity      | 15%    |
| Rewrite Quality    | 15%    |

---

### Run Evaluations

```bash
python -m evals.runner
```

---

## 📊 Severity Scale

| Score | Meaning        |
| ----- | -------------- |
| 1–2   | Mild           |
| 3–4   | Bias           |
| 5–6   | Explicit hate  |
| 7–8   | Dehumanization |
| 9–10  | Violence       |

---

## ✨ Features

* Multi-agent AI pipeline
* Structured JSON outputs
* Hate speech classification
* Severity scoring
* Respectful rewriting
* Evaluation framework
* Dataset support
* Retry handling
* Web UI

---

## 🧪 Example

**Input**

```text
People from that country are all criminals.
```

**Output**

```json
{
  "verdict": "unsafe",
  "category": "xenophobia",
  "severity": 7,
  "rewritten_text": "We should avoid generalizing people based on nationality."
}
```

---

## 🔧 Customization

* Change model → `agents.py`
* Modify prompts → `tasks.py`
* Extend dataset → `evals/dataset.json`
* Add new agents

---

## 📂 Project Structure

```text
hate-speech-guardian/
├── app.py
├── agents.py
├── tasks.py
├── evals/
├── templates/
├── assets/            # 👈 add architecture image here
├── .env.example
├── requirements.txt
└── README.md
```

---

## 🧠 Key Learnings

* Multi-agent systems improve reliability
* Structured outputs reduce failures
* Evaluation is critical
* Model choice > prompt length
* Rate limits require handling

---

## 📜 License

MIT License

---

## ⭐ Support

If you found this useful:

* ⭐ Star the repo
* 🔁 Share it
* 🛠️ Build on top of it

---

## 🙌 Contributing

Pull requests are welcome!

