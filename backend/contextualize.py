# rewrites a follow-up question into a standalone question before retrieval, so that retrieval has the full context of what is being asked previously
# important for follow up questions because it rewrites pronouns and vague references into explicit references

import re
import groq
from backend.config import CONTEXTUALIZE_MODEL
from backend.debug_log import log
from backend.generate import iter_groq_clients, advance_groq_key

_AMBIGUITY_SIGNALS = re.compile(
    r"\b(it|this|that|these|those|they|them|he|she|same|also|instead|"
    r"what about|how about)\b",
    re.IGNORECASE,
)
_STANDALONE_MIN_WORDS = 7

TOPIC_ANCHORED_SYSTEM = (
    "The user is asking a follow-up question about a SPECIFIC topic, given "
    "below. Rewrite the follow-up into a standalone question that names "
    "that topic explicitly, replacing vague words like 'it', 'that', "
    "'this one' with the topic name. Keep the user's wording and intent "
    "otherwise unchanged. Do NOT answer the question -- only rewrite it. "
    "Output ONLY the rewritten question, nothing else.\n\n"
    "Example:\n"
    "CURRENT TOPIC: Application for New Business Permit (Onsite)\n"
    "FOLLOW-UP QUESTION: how long?\n"
    "STANDALONE QUESTION: How long does it take to process an Application "
    "for New Business Permit (Onsite)?"
)

GENERAL_SYSTEM = (
    "Given a conversation history and a follow-up question, rewrite the "
    "follow-up question into a standalone question that contains all the "
    "context needed to understand it on its own, without reading the "
    "history. Preserve the user's original intent and wording as much as "
    "possible -- only add what's missing (e.g. replace 'it', 'that', "
    "'this one' with the actual subject). "
    "If the follow-up question is already standalone and doesn't depend on "
    "the history, return it completely unchanged. "
    "Do NOT answer the question. Do NOT add explanations, quotes, or any "
    "text other than the rewritten question itself."
)


def _last_topic(history: list[dict]) -> str | None:
    for turn in reversed(history):
        if turn.get("role") == "assistant" and turn.get("topic"):
            return turn["topic"]
    return None


def _format_history(history: list[dict], max_turns: int = 4) -> str:
    recent = history[-max_turns:]
    return "\n".join(
        f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
        for turn in recent
    )


def _call_groq(system: str, prompt: str) -> str | None:
    try:
        for key_idx, client in iter_groq_clients(timeout=20.0):
            try:
                completion = client.chat.completions.create(
                    model=CONTEXTUALIZE_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_tokens=300,
                    extra_body={"reasoning_format": "hidden"},
                )
                rewritten = (completion.choices[0].message.content or "").strip()
                return rewritten.strip('"').strip("'") or None
            except groq.RateLimitError:
                advance_groq_key(key_idx)
                continue
        return None
    except Exception as e:
        # Groq down/slow/rate-limited/etc: don't break the chat, just skip rewriting; but log why it happened
        log(f"[contextualize] _call_groq failed, falling back to raw question: {e!r}")
        return None


def contextualize_question(question: str, history: list[dict] | None = None) -> str:
    history = history or []
    if not history:
        return question

    # pre-filter to skip the LLM call for questions that are long enough; these are almost already standalone
    word_count = len(question.split())
    if word_count >= _STANDALONE_MIN_WORDS and not _AMBIGUITY_SIGNALS.search(question):
        return question

    topic = _last_topic(history)
    if topic:
        prompt = (
            f"CURRENT TOPIC: {topic}\n"
            f"FOLLOW-UP QUESTION: {question}\n"
            f"STANDALONE QUESTION:"
        )
        rewritten = _call_groq(TOPIC_ANCHORED_SYSTEM, prompt)
    else:
        history_block = _format_history(history)
        prompt = (
            f"CONVERSATION HISTORY:\n{history_block}\n\n"
            f"FOLLOW-UP QUESTION:\n{question}\n\n"
            f"STANDALONE QUESTION:"
        )
        rewritten = _call_groq(GENERAL_SYSTEM, prompt)

    result = rewritten if rewritten else question
    log(f"[contextualize] topic_anchor={topic!r} question={question!r} -> search_query={result!r}")
    return result