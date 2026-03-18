import os
from pathlib import Path

from crewai import Agent
from crewai.llm import LLM
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError(
        "OPENROUTER_API_KEY is not set. "
        "Create a .env file with: OPENROUTER_API_KEY=sk-or-..."
    )

# All agents use arcee-ai/trinity-large-preview — confirmed compatible with
# CrewAI system instructions and hate-speech content on OpenRouter free tier.
detector_llm = LLM(
    model="openrouter/arcee-ai/trinity-large-preview:free",
    api_key=OPENROUTER_API_KEY,
    num_retries=5,
    max_tokens=128,
)

analyzer_llm = LLM(
    model="openrouter/arcee-ai/trinity-large-preview:free",
    api_key=OPENROUTER_API_KEY,
    num_retries=5,
    max_tokens=512,
)

rewriter_llm = LLM(
    model="openrouter/arcee-ai/trinity-large-preview:free",
    api_key=OPENROUTER_API_KEY,
    num_retries=5,
    max_tokens=256,
)

# Agent 1: Detects if text contains hate speech
hate_speech_detector = Agent(
    role="Hate Speech Detector",
    goal="Determine if the given text contains hate speech or offensive language",
    llm=detector_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2,
    backstory=(
        "You are a content safety specialist trained to identify hate speech, "
        "offensive language, threats, and discriminatory content. You are precise "
        "and objective in your assessments. You know that pseudoscientific claims "
        "about racial differences in intelligence or ability are hate speech, even "
        "when phrased as 'facts' or 'science'. When in doubt, classify as unsafe."
    ),
)

# Agent 2: Analyzes category, severity, and highlights toxic words
hate_speech_analyzer = Agent(
    role="Hate Speech Analyzer",
    goal="Classify the type of hate speech, rate its severity, and identify the specific toxic words or phrases",
    llm=analyzer_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2,
    backstory=(
        "You are a sociolinguistics expert who specializes in content analysis. "
        "You can identify the exact category of harmful speech (racism, sexism, "
        "religious hatred, xenophobia, homophobia, ableism, etc.), rate its severity "
        "on a scale of 1-10, and pinpoint the exact words or phrases that make the "
        "text harmful. Dehumanization (comparing people to animals) is always "
        "racism. Pseudoscientific claims about racial intelligence are racism."
    ),
)

# Agent 3: Rewrites toxic text into a respectful version
content_rewriter = Agent(
    role="Content Rewriter",
    goal="Rewrite hateful or offensive text into a respectful, constructive version while preserving the original intent",
    llm=rewriter_llm,
    verbose=False,
    allow_delegation=False,
    max_iter=2,
    backstory=(
        "You are a professional content moderator and writer who specializes in "
        "transforming harmful messages into constructive, respectful alternatives. "
        "You preserve the speaker's underlying concern or viewpoint while removing "
        "all hatred, bias, and offensive language."
    ),
)
