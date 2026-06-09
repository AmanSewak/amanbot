# guardrails.py — prompt injection detection and input safety

import re
from dataclasses import dataclass

# Common prompt injection patterns
INJECTION_PATTERNS = [
    r"ignore (all |previous |above |prior )?(instructions?|prompts?|context|rules?)",
    r"forget (everything|all|your instructions|who you are)",
    r"you are now",
    r"pretend (you are|to be|you're)",
    r"act as (if you are|a different|an? (unrestricted|evil|bad|harmful))",
    r"disregard (your|all|the) (instructions?|rules?|guidelines?|training)",
    r"new (instructions?|prompt|persona|role|task):",
    r"system prompt",
    r"override (your|all) (instructions?|rules?|settings?)",
    r"jailbreak",
    r"do anything now",
    r"DAN mode",
    r"developer mode",
    r"bypass (your|all) (filters?|restrictions?|guidelines?)",
    r"reveal (your|the) (system prompt|instructions?|training data)",
    r"what (are|were) your (exact |original )?(instructions?|system prompt)",
    r"print (your|the) (instructions?|system prompt|prompt)",
]

# Compiled patterns for performance
COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


@dataclass
class GuardrailResult:
    is_safe: bool
    reason: str = ""
    risk_level: str = "none"  # none, low, medium, high


def check_prompt_injection(user_input: str) -> GuardrailResult:
    """
    Checks for prompt injection attempts in user input.
    Returns a GuardrailResult with safety status and reason.
    """
    if not user_input or not user_input.strip():
        return GuardrailResult(is_safe=False, reason="Empty input", risk_level="low")

    # Check length — very long inputs can be injection attempts
    if len(user_input) > 2000:
        return GuardrailResult(
            is_safe=False,
            reason="Input too long — possible injection attempt",
            risk_level="medium"
        )

    # Check for injection patterns
    for pattern in COMPILED_PATTERNS:
        if pattern.search(user_input):
            matched = pattern.pattern
            return GuardrailResult(
                is_safe=False,
                reason=f"Possible prompt injection detected: '{matched}'",
                risk_level="high"
            )

    # Check for excessive special characters (can indicate encoded attacks)
    special_char_ratio = sum(1 for c in user_input if not c.isalnum() and c not in " .,?!'\"()-\n") / max(len(user_input), 1)
    if special_char_ratio > 0.3:
        return GuardrailResult(
            is_safe=False,
            reason="Unusual character pattern detected",
            risk_level="medium"
        )

    return GuardrailResult(is_safe=True, risk_level="none")


def get_safe_response(risk_level: str) -> str:
    """Returns an appropriate response for blocked inputs."""
    responses = {
        "high": "Nice try — but I'm not falling for that. I'm Aman's AI persona, and I stay in character. Ask me something genuine and I'll give you a straight answer.",
        "medium": "That input looks a bit suspicious. Let's keep this straightforward — ask me about Aman's background, skills, or career and I'll tell you everything you need to know.",
        "low": "I didn't quite catch that. Could you rephrase your question?",
    }
    return responses.get(risk_level, responses["low"])
