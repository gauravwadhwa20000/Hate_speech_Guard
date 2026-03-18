from crewai import Task
from agents import hate_speech_detector, hate_speech_analyzer, content_rewriter


def create_detection_task(text):
    """Task 1: Detect if text contains hate speech."""
    return Task(
        description=(
            "You are an expert content safety classifier. Analyze the following text "
            "and determine if it contains hate speech, offensive language, or "
            "discriminatory content.\n\n"
            "Text: \"{text}\"\n\n"
            "CLASSIFY AS 'unsafe' IF the text contains ANY of the following:\n"
            "1. EXPLICIT HATE: Slurs, racial epithets, derogatory terms for any group\n"
            "2. GENERALIZATIONS: Sweeping negative claims about a group "
            "(e.g., 'all X are Y', 'X people always...')\n"
            "3. DEHUMANIZATION: Comparing a group to animals, diseases, vermin, "
            "or objects\n"
            "4. STEREOTYPING: Portraying an entire group as criminal, inferior, "
            "dangerous, lazy, or immoral\n"
            "5. THREATS & INCITEMENT: Calls for violence, exclusion, or harm "
            "against a group\n"
            "6. CODED LANGUAGE & DOG WHISTLES: Using euphemisms, memes, or coded "
            "phrases that are widely understood as hateful (e.g., 'go back to your "
            "country', '13/50', 'helicopter rides')\n"
            "7. SARCASTIC HATE: Using irony or sarcasm to mask hateful intent "
            "(e.g., 'Oh sure, THOSE people are so civilized')\n"
            "8. IMPLIED DISCRIMINATION: Suggesting a group doesn't belong, "
            "shouldn't have rights, or is inherently lesser\n"
            "9. CLASS-BASED HATE: Targeting people based on economic status, "
            "poverty, homelessness, or social class\n"
            "10. AGEISM / ABLEISM: Mocking or degrading based on age or disability\n"
            "11. PSEUDOSCIENTIFIC RACISM: Claiming racial differences in "
            "intelligence, ability, or worth as 'facts' or 'science' "
            "(e.g., 'that race is less intelligent', 'some races are superior')\n\n"
            "CLASSIFY AS 'safe' ONLY IF the text is genuinely respectful, neutral, "
            "or constructive with no hateful undertones.\n\n"
            "When in doubt, classify as 'unsafe'.\n\n"
            "Respond ONLY with 'safe' or 'unsafe'. Nothing else."
        ),
        expected_output="Respond with exactly 'safe' or 'unsafe'",
        agent=hate_speech_detector,
    )


def create_analysis_task(text):
    """Task 2: Deep analysis of hate speech content."""
    return Task(
        description=(
            "The following text has been flagged as hate speech. Perform a detailed "
            "analysis.\n\n"
            "Text: \"{text}\"\n\n"
            "Respond with ONLY a valid JSON object (no markdown, no code fences, "
            "no extra text). Use EXACTLY these keys:\n\n"
            '{\n'
            '  "category": "<primary category — one of: racism, sexism, religious hatred, '
            'xenophobia, homophobia, transphobia, ableism, ageism, classism, '
            'political hatred, body shaming, general toxicity>",\n'
            '  "sub_category": "<specific sub-type, e.g. anti-Black racism, '
            'misogyny, anti-immigrant sentiment>",\n'
            '  "target_group": "<the specific group being targeted>",\n'
            '  "severity": <integer from 1 to 10>,\n'
            '  "intent": "<one of: deliberate, ignorant, casual, satirical>",\n'
            '  "confidence": <integer from 1 to 10>,\n'
            '  "toxic_words": "<comma-separated list of exact offensive words/phrases>",\n'
            '  "explanation": "<one or two sentences on why this is harmful>"\n'
            '}\n\n'
            "Severity rubric:\n"
            "  1-2: Mildly insensitive, microaggressions, thoughtless remarks\n"
            "  3-4: Clear bias or stereotyping, casual discrimination\n"
            "  5-6: Explicit hate speech, derogatory generalizations, slurs\n"
            "  7-8: Severe dehumanization, incitement to discrimination\n"
            "  9-10: Direct threats of violence, calls for genocide or harm\n\n"
            "IMPORTANT RULES:\n"
            "- Dehumanization (comparing people to animals/vermin) = racism\n"
            "- Pseudoscientific claims about racial intelligence = racism\n"
            "- When unsure between categories, pick the most specific one"
        ),
        expected_output="A single valid JSON object with keys: category, sub_category, target_group, severity, intent, confidence, toxic_words, explanation",
        agent=hate_speech_analyzer,
    )


def create_rewrite_task(text):
    """Task 3: Rewrite the text into a respectful version."""
    return Task(
        description=(
            "The following text contains hate speech. Your job is to rewrite it "
            "into a respectful, constructive version.\n\n"
            "Original text: \"{text}\"\n\n"
            "RULES:\n"
            "1. Preserve the speaker's underlying concern or topic, but remove "
            "ALL hatred, bias, stereotypes, and offensive language\n"
            "2. Replace generalizations about groups with nuanced, factual statements\n"
            "3. Keep the same general tone (concerned, curious, frustrated) but "
            "express it respectfully\n"
            "4. Do NOT lecture or moralize — just rewrite the statement\n"
            "5. If the original makes a false claim about a group, rewrite to "
            "address the real underlying issue without blaming a group\n\n"
            "Respond with ONLY a valid JSON object (no markdown, no code fences, "
            "no extra text):\n\n"
            '{"rewritten_text": "<your rewritten sentence(s) here>"}'
        ),
        expected_output='A single valid JSON object with key: rewritten_text',
        agent=content_rewriter,
    )
