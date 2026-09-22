import requests
from backend.config import OLLAMA_BASE_URL, OLLAMA_MODEL

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
    "do not abbreviate or cut off enumerations."
)


def build_prompt(question: str, retrieved_chunks: list[dict], history: list[dict] | None = None) -> str:
    context_block = "\n\n".join(
        f"[{c['title']} | category: {c['category']}]\n{c['text']}"
        for c in retrieved_chunks
    ) or "No relevant context was found."

    history_block = ""
    if history:
        turns = "\n".join(
            f"{'User' if turn['role'] == 'user' else 'Assistant'}: {turn['content']}"
            for turn in history[-6:]
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


def generate_answer(question: str, retrieved_chunks: list[dict], history: list[dict] | None = None) -> str:
    prompt = build_prompt(question, retrieved_chunks, history)

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "system": SYSTEM_INSTRUCTION,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 8192,
                    "num_predict": 768,
                },
            },
            timeout=120,
        )
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not reach Ollama at {OLLAMA_BASE_URL}. Make sure Ollama is "
            f"running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {OLLAMA_MODEL}`)."
        )

    if response.status_code == 404:
        raise RuntimeError(
            f"Ollama can't find model '{OLLAMA_MODEL}'. Run: ollama pull {OLLAMA_MODEL}"
        )
    response.raise_for_status()

    data = response.json()
    return data["response"].strip()