# Hate Speech Guardian

![GitHub stars](https://img.shields.io/github/stars/gauravwadhwa20000/hate-speech-guardian?style=social)
![GitHub forks](https://img.shields.io/github/forks/gauravwadhwa20000/hate-speech-guardian?style=social)
![GitHub issues](https://img.shields.io/github/issues/gauravwadhwa20000/hate-speech-guardian)
![License](https://img.shields.io/github/license/gauravwadhwa20000/hate-speech-guardian)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![Built with CrewAI](https://img.shields.io/badge/Built%20With-CrewAI-purple)
![LLM](https://img.shields.io/badge/Powered%20By-OpenRouter-green)

A multi-agent AI content moderation pipeline built with **CrewAI** and **Flask**.

Three specialized AI agents collaborate to:
- 🔍 Detect hate speech  
- 🧠 Analyze intent, category, and severity  
- ✨ Rewrite it into respectful language  

All powered by **free LLMs via OpenRouter**.

---

## 🎥 Demo

👉 https://www.youtube.com/watch?v=xNqT8rf-5Jc

---

## 🔥 Why This Project Stands Out

- 🤖 **Multi-Agent AI System** (not just prompts)
- 📊 **Evaluation-driven development**
- 🧠 **Structured JSON outputs (production-ready)**
- 💸 **Runs on FREE LLMs**
- ⚡ **Optimized pipeline (short-circuit + caching)**
- 🧪 **50 test cases + deep diagnostics**

---

## 📚 Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Model](#model)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Evaluations](#evaluations)
- [Key Learnings](#key-learnings)
- [Example Inputs](#example-inputs)
- [License](#license)

---

## 🚀 Project Overview

Online content moderation is hard:
- Manual review → slow & expensive  
- Single LLM → misses nuance  

This project solves it using **3 specialized AI agents**:

1. **Detector** → Safe vs Unsafe  
2. **Analyzer** → Category, severity, intent  
3. **Rewriter** → Respectful transformation  

Plus:
- Full **evaluation framework**
- **Per-agent diagnostics**
- **Dataset editor + reports**

---

## 🏗 Architecture

```mermaid
flowchart TD
    A[User Input - Web UI] --> B[Flask Backend]
    B --> C[Agent 1: Detector]

    C -->|Safe| H[Return Response]
    C -->|Unsafe| D[Agent 2: Analyzer]

    D --> E[Structured JSON]
    E --> F[Agent 3: Rewriter]
    F --> G[Respectful Output]

    G --> H[Final Response]

    subgraph LLM Layer (OpenRouter)
        C
        D
        F
    end
````

---

## ⚙️ How It Works

```
User Input → Detector → Analyzer → Rewriter → UI
```

* If **safe** → pipeline stops early
* If **unsafe** → full pipeline executes

---

## 🧠 Model

All agents use:

```
arcee-ai/trinity-large-preview:free
```

| Agent    | Role                | Strength        |
| -------- | ------------------- | --------------- |
| Detector | Classification      | Fast + accurate |
| Analyzer | JSON structuring    | Reliable output |
| Rewriter | Text transformation | Clean rewrites  |

---

## ✨ Features

### 🔹 Pipeline

* Multi-agent orchestration (CrewAI)
* 11-rule hate detection
* Category classification
* Severity scoring (1–10)
* Intent + confidence detection
* Toxic word highlighting
* Respectful rewriting
* Retry with exponential backoff

### 🔹 Evaluation System

* 50 labeled test cases
* Weighted scoring system
* Per-agent deep analysis
* Dataset editor (UI)
* Downloadable reports
* Disk caching
* Quick run + rerun failures

---

## 🛠 Tech Stack

| Layer        | Tool         |
| ------------ | ------------ |
| AI Framework | CrewAI       |
| LLM Routing  | LiteLLM      |
| LLM Gateway  | OpenRouter   |
| Backend      | Flask        |
| Language     | Python 3.11+ |
| Frontend     | HTML/CSS/JS  |

---

## 📁 Project Structure

```
hate-speech-guardian/
├── app.py
├── agents.py
├── tasks.py
├── evals/
│   ├── runner.py
│   ├── dataset.json
│   └── cache.json
├── templates/
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone Repo

```bash
git clone https://github.com/gauravwadhwa20000/hate-speech-guardian.git
cd hate-speech-guardian
```

### 2. Setup Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Add API Key

```
OPENROUTER_API_KEY=your_key
```

### 4. Run App

```bash
python app.py
```

👉 Open: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🖥 Usage

### Analyze Tab

* Input text
* View detection, analysis, rewrite

### Evals Tab

* Run test cases
* View accuracy + diagnostics

### Dataset Tab

* Edit / add / delete cases

---

## 📈 Evaluations

| Metric   | Weight |
| -------- | ------ |
| Verdict  | 30%    |
| Category | 25%    |
| Severity | 15%    |
| JSON     | 15%    |
| Rewrite  | 15%    |

Run:

```bash
python -m evals.runner
python -m evals.runner --quick
```

---

## 🧠 Key Learnings

* Multi-agent systems need optimization
* Evals > intuition
* Model choice > prompt size
* JSON > regex
* Caching is critical
* Per-agent debugging is powerful

---

## 🧪 Example Inputs

| Input        | Output                    |
| ------------ | ------------------------- |
| Neutral text | ✅ Safe                    |
| Hate speech  | ❌ Categorized + rewritten |

---

## ⭐ Support

If you find this useful:

* Star ⭐ the repo
* Fork 🍴 and experiment
* Share 🚀 with AI builders

---

## 📄 License

MIT License

```

---

# 🔥 What You Now Have

This version is:
- ✅ Clean + structured  
- ✅ Recruiter-friendly  
- ✅ Developer-friendly  
- ✅ Viral-ready  
- ✅ Production-grade  

---

If you want next step (high impact):
👉 I can turn this into a **LinkedIn viral post + thumbnail + caption strategy** that drives traffic to your repo.
```
