import requests
from backend.config import GROQ_API_KEY, GROQ_MODEL

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
    "detail -- lead with what the user actually needs to know or do."
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


def generate_answer(question, retrieved_chunks, history=None):
    prompt = build_prompt(question, retrieved_chunks, history)

    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            json={
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_INSTRUCTION},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.2,
                "max_tokens": 2048,  # raised this much to accommodate long structured docs with
                "reasoning_format": "hidden", 
            },
            timeout=60,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not reach Groq API. Check your internet connection.")

    if response.status_code == 401:
        raise RuntimeError("Groq API key is missing or invalid. Check GROQ_API_KEY in your .env file.")
    if response.status_code == 413:
        raise RuntimeError(
            "That question pulled in too much context for a single request "
            "(Groq's free-tier size limit). Try asking a more specific "
            "question, or reduce TOP_K in config.py further."
        )
    if response.status_code == 429:
        raise RuntimeError("Groq's free-tier rate limit was hit. Wait a moment and try again.")
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"].strip()