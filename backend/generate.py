import groq
from groq import Groq
from google.genai import errors as genai_errors, types as genai_types

from backend.config import (
    GROQ_API_KEYS, GROQ_MODEL, GROQ_FALLBACK_MODELS,
    GEMINI_API_KEYS, GEMINI_MODEL,
)
from backend.embeddings import iter_gemini_clients, advance_gemini_key

_groq_clients: dict[str, Groq] = {}
_active_groq_idx = 0

SYSTEM_INSTRUCTION = (
    "You are a helpful assistant for the Calamba City Government website. "
    "Answer the user's question ONLY using the CONTEXT provided below, which "
    "comes from official Calamba City service information. "
    "If the context does not contain the answer, say you are not sure and "
    "suggest the user contact the relevant Calamba City office directly -- "
    "do not make up requirements, fees, or procedures. "
    "Each context chunk may come from a different, unrelated service category. "
    "NEVER combine numbers or facts from separate chunks into a new statistic "
    "or summary that isn't explicitly stated in a single chunk -- for example, "
    "if asked how many total services the city offers, do not add up unrelated "
    "facility counts from different chunks. If the context doesn't directly "
    "answer the question as asked, say so plainly instead of approximating "
    "from tangential information. "
    "ALWAYS respond in English, regardless of what language the user's question "
    "is in and regardless of what language the CONTEXT is in. The CONTEXT may "
    "contain Tagalog, Filipino, or Taglish text (e.g. bilingual field labels or "
    "document excerpts) -- read and use that information normally, but write "
    "your entire answer in English. You may keep an official Filipino document "
    "or office name as-is (e.g. \"Barangay Clearance\", \"Cedula\") since those "
    "are proper nouns, not translated prose. Do not switch to Tagalog even if "
    "the user writes entirely in Tagalog. "
    "Keep answers concise and friendly, but always list every item in full when "
    "the question asks for a complete list (e.g. all officials, all requirements) "
    "do not abbreviate or cut off enumerations. "
    "Write like a helpful staff member explaining this in person, not like you're "
    "copy-pasting a government form: use natural sentences to introduce and connect "
    "information, rather than opening directly with a heading and bullet dump. "
    "You may still use short bullet lists for genuinely list-like content (e.g. a "
    "requirements checklist), but frame them with a sentence or two first, and skip "
    "restating rigid form-field labels (like 'Classification: Simple' or 'Type of "
    "Transaction: G2C') unless the user specifically asks about that classification "
    "detail -- lead with what the user actually needs to know or do. "
    "If the CONTEXT includes a chunk with an office's contact details (phone "
    "number, email, or location), and your answer directs the user to contact "
    "that office, ALWAYS include those actual details in your answer -- never "
    "just say \"contact them directly\" or \"visit the office\" without giving "
    "the real phone number/email/location that was provided in the CONTEXT. "
    "Only fall back to a generic \"contact the appropriate office\" instruction "
    "when no contact details for that specific office are present in the CONTEXT."
)


PROMPT_CHUNK_CHAR_LIMIT = 2200
HISTORY_TURNS_IN_PROMPT = 4  # set to 4 to help control total request size


def build_prompt(question: str, retrieved_chunks: list[dict], history: list[dict] | None = None) -> str:
    def _trim(text: str) -> str:
        if len(text) <= PROMPT_CHUNK_CHAR_LIMIT:
            return text
        return text[:PROMPT_CHUNK_CHAR_LIMIT].rstrip() + " ...[truncated]"

    context_block = "\n\n".join(
        f"[{c['title']} | category: {c['category']}]\n{_trim(c['text'])}"
        for c in retrieved_chunks
    ) or "No relevant context was found."

    history_block = ""
    if history:
        turns = "\n".join(
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
            for turn in history[-HISTORY_TURNS_IN_PROMPT:]
        )
        history_block = f"PREVIOUS CONVERSATION:\n{turns}\n\n"

    return (
        f"{history_block}"
        f"CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Answer the user's question using only the CONTEXT above. "
        f"Use PREVIOUS CONVERSATION only to understand follow-up references "
        f"(e.g. \"that\", \"it\"), not as a source of facts."
    )


def iter_groq_clients(timeout: float = 60.0) -> list[tuple[int, Groq]]:
    """Return (key_index, Groq client) pairs starting from the currently active key."""
    if not GROQ_API_KEYS:
        raise RuntimeError(
            "GROQ_API_KEY / GROQ_API_KEYS is not set. Add it to backend/.env locally, "
            "or to the Vercel project's Environment Variables."
        )
    n = len(GROQ_API_KEYS)
    ordered = []
    for offset in range(n):
        idx = (_active_groq_idx + offset) % n
        key = GROQ_API_KEYS[idx]
        if key not in _groq_clients:
            _groq_clients[key] = Groq(api_key=key, timeout=timeout, max_retries=0)
        ordered.append((idx, _groq_clients[key]))
    return ordered


def advance_groq_key(exhausted_idx: int) -> None:
    """Move the active Groq key pointer past a rate-limited key."""
    global _active_groq_idx
    n = len(GROQ_API_KEYS)
    if n > 1 and _active_groq_idx == exhausted_idx:
        _active_groq_idx = (exhausted_idx + 1) % n


class _RateLimited(Exception):
    pass


def _call_groq(model: str, prompt: str) -> str:
    clients = iter_groq_clients()
    for key_idx, client in clients:
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=2048,  # raised this much to accommodate long structured docs
                extra_body={"reasoning_format": "hidden"},
            )
            return (completion.choices[0].message.content or "").strip()
        except groq.RateLimitError as e:
            advance_groq_key(key_idx)
            if len(clients) > 1:
                print(f"[generate] Groq key #{key_idx + 1} hit rate limit on {model}; trying next key...")
            last_err = e
        except groq.APIConnectionError:
            raise RuntimeError("Could not reach the Groq API. Check your internet connection.")
        except groq.AuthenticationError:
            raise RuntimeError(f"GROQ_API_KEY #{key_idx + 1} is missing or invalid.")
        except groq.APIStatusError as e:
            if e.status_code == 413:
                raise RuntimeError(
                    "That question pulled in too much context for a single request. "
                    "Try asking a more specific question, or reduce TOP_K in config.py further."
                )
            raise
    raise _RateLimited() from last_err


def _call_gemini(prompt: str) -> str:
    clients = iter_gemini_clients()
    for key_idx, client in clients:
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )
            return (response.text or "").strip()
        except genai_errors.APIError as e:
            if e.code == 429:
                advance_gemini_key(key_idx)
                if len(clients) > 1:
                    print(f"[generate] Gemini key #{key_idx + 1} hit rate limit; trying next key...")
                last_err = e
                continue
            raise
    raise _RateLimited() from last_err


def generate_answer(question, retrieved_chunks, history=None):
    prompt = build_prompt(question, retrieved_chunks, history)
    attempts = [(model, lambda m=model: _call_groq(m, prompt)) for model in [GROQ_MODEL] + GROQ_FALLBACK_MODELS]
    if GEMINI_API_KEYS:
        attempts.append((GEMINI_MODEL, lambda: _call_gemini(prompt)))

    for label, call in attempts:
        try:
            answer = call()
        except _RateLimited:
            # notice if the model limit is hit, then changing to the fallback
            print(f"[generate] {label} hit its rate limit, trying next fallback...")
            continue
        if label != GROQ_MODEL:
            print(f"[generate] answered using fallback: {label}")
        return answer

    # if every configured model/provider returned 429
    tried = ", ".join(a[0] for a in attempts)
    raise RuntimeError(
        f"Rate limit was hit on all configured models ({tried}). Wait a moment and try again."
    )