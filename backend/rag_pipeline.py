import re
import json

import numpy as np

from backend.preprocess import clean_text, is_greeting, correct_typos, build_domain_vocabulary
from backend.intent_classifier import predict_intent
from backend.embeddings import embed_query
from backend.generate import generate_answer
from backend.contextualize import contextualize_question
from backend.debug_log import log
from backend.forms_lookup import get_forms_for_office, get_all_forms
from backend.config import (
    TOP_K, CATEGORIES, FORMS_MIN_SIMILARITY, MIN_SIMILARITY, DOCUMENTS_PATH,
    VECTOR_STORE_PATH, CHUNKS_META_PATH, EMBEDDING_DIM,
)
from backend.office_directory import build_office_lookup, find_office_doc_id


# ---------------------------------------------------------------------------
# static vector store: loaded into memory ONCE at module import (cold start),
# never written to at runtime (Vercel's filesystem is read-only)
# ---------------------------------------------------------------------------

def _load_vector_store():
    if not VECTOR_STORE_PATH.is_file() or not CHUNKS_META_PATH.is_file():
        raise RuntimeError(
            f"Static vector store not found ({VECTOR_STORE_PATH.name} / {CHUNKS_META_PATH.name}). "
            "Run `python -m backend.build_index` locally and commit the generated files."
        )
    with np.load(VECTOR_STORE_PATH) as npz:
        embeddings = np.ascontiguousarray(npz["embeddings"], dtype=np.float32)
    with open(CHUNKS_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    chunks = meta["chunks"]
    if embeddings.shape != (len(chunks), EMBEDDING_DIM):
        raise RuntimeError(
            f"Vector store shape {embeddings.shape} does not match {len(chunks)} chunks x "
            f"{EMBEDDING_DIM} dims. Re-run `python -m backend.build_index`."
        )
    return embeddings, chunks


_EMBEDDINGS, _CHUNKS = _load_vector_store()
_CHUNK_INDEX = {c["chunk_id"]: i for i, c in enumerate(_CHUNKS)}
_CATEGORY_ARR = np.array([c["category"] for c in _CHUNKS])
_DOC_ID_ARR = np.array([c["doc_id"] for c in _CHUNKS])


def _hit(i: int, similarity: float | None) -> dict:
    c = _CHUNKS[i]
    return {
        "chunk_id": c["chunk_id"],
        "text": c["text"],
        "title": c["title"],
        "category": c["category"],
        "office": c.get("office") or None,
        "doc_id": c["doc_id"],
        "similarity": similarity,
    }


def retrieve(question: str, top_k: int = 3, category_filter: str | None = None) -> list[dict]:
    # embed the question, rank chunks by cosine similarity (dot product of unit vectors), drop hits below MIN_SIMILARITY
    query_vector = embed_query(question)
    scores = _EMBEDDINGS @ query_vector

    if category_filter:
        candidates = np.flatnonzero(_CATEGORY_ARR == category_filter)
    else:
        candidates = np.arange(len(_CHUNKS))
    if candidates.size == 0:
        return []

    cand_scores = scores[candidates]
    k = min(top_k, candidates.size)
    top = np.argpartition(-cand_scores, k - 1)[:k]
    top = top[np.argsort(-cand_scores[top])]

    hits = []
    for j in top:
        similarity = float(cand_scores[j])
        if similarity < MIN_SIMILARITY:
            continue
        hits.append(_hit(int(candidates[j]), similarity))
    return hits


def get_chunks_by_doc_id(doc_id: str) -> list[dict]:
    # every chunk of one document, bypassing similarity search (used when the document is already known)
    return [_hit(int(i), None) for i in np.flatnonzero(_DOC_ID_ARR == doc_id)]


def get_chunk_by_id(chunk_id: str) -> dict | None:
    # one specific chunk by id, e.g. a document's summary chunk or an office contact chunk
    i = _CHUNK_INDEX.get(chunk_id)
    return _hit(i, None) if i is not None else None


def retrieve_all_in_category(category: str) -> list[dict]:
    # every chunk tagged with this category, no ranking, no top_k cap (enumeration queries)
    return [_hit(int(i), None) for i in np.flatnonzero(_CATEGORY_ARR == category)]


_vocab = build_domain_vocabulary(_CHUNKS)

_office_lookup_cache = None


def _get_office_lookup() -> dict:
    global _office_lookup_cache
    if _office_lookup_cache is None:
        with open(DOCUMENTS_PATH, "r", encoding="utf-8") as f:
            _office_lookup_cache = build_office_lookup(json.load(f))
    return _office_lookup_cache


def _attach_office_contact(retrieved: list[dict], office: str | None) -> list[dict]:
    if not office:
        return retrieved
    office_doc_id = find_office_doc_id(office, _get_office_lookup())
    if not office_doc_id:
        return retrieved
    office_chunk_id = f"{office_doc_id}_chunk0"
    if any(c.get("chunk_id") == office_chunk_id for c in retrieved):
        return retrieved
    office_chunk = get_chunk_by_id(office_chunk_id)
    if not office_chunk:
        return retrieved
    log(f"[retrieval] attached office contact info: {office!r} -> {office_chunk_id!r}")
    return retrieved + [office_chunk]


GREETING_RESPONSE = (
    "Hello! I'm the virtual assistant for the City Government of Calamba. "
    "I can help with questions about certificates & permits, business & trade, "
    "health, education, tax, housing, disaster preparedness, and other city "
    "services. What would you like to know?"
)

ENUMERATION_PATTERNS = re.compile(
    r"\b(all|list|every|what are the)\b.*\b(services?|programs?)\b",
    re.IGNORECASE,
)

META_QUERY_PATTERN = re.compile(
    r"\bhow many (services|programs|offices|departments)\b", re.IGNORECASE
)

FORMS_AVAILABILITY_PATTERN = re.compile(
    r"\b(other|additional|more|else)\b.{0,60}\b(downloadable\s+)?(forms?|files?|documents?)\b"
    r"|\b(downloadable\s+)?(forms?|files?)\b.{0,60}\b(other|additional|more|else)\b"
    r"|\ball\s+(the\s+)?(downloadable\s+)?(forms?|files?)\b"
    r"|\blist\b.{0,40}\b(downloadable\s+)?(forms?|files?)\b"
    r"|\bwhat\s+(downloadable\s+)?(forms?|files?)\s+(are\s+)?(available|do you have|can i download)\b",
    re.IGNORECASE,
)

NO_MATCHING_FORMS_ANSWER = (
    "I don't have any downloadable forms on file right now."
)

REFUSAL_PATTERN = re.compile(
    r"\b(not sure|does not contain|doesn'?t contain|no information (about|on|regarding))\b",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_PATTERN.search(answer))


def is_enumeration_query(question: str) -> bool:
    return bool(ENUMERATION_PATTERNS.search(question))


def is_meta_query(question: str) -> bool:
    return bool(META_QUERY_PATTERN.search(question))


def is_forms_availability_query(question: str) -> bool:
    return bool(FORMS_AVAILABILITY_PATTERN.search(question))


def answer_question(
    question: str,
    history: list[dict] | None = None,
    category_hint: str | None = None,
    doc_id_hint: str | None = None,
) -> dict:
    question = clean_text(question)
    history = history or []

    question = correct_typos(question, _vocab)

    if is_greeting(question):
        return {
            "question": question,
            "predicted_intent": "GREETING",
            "intent_label": "Greeting",
            "intent_confidence": 1.0,
            "retrieved_chunks": [],
            "answer": GREETING_RESPONSE,
            "forms": [],
            "history": history,
        }

    if is_meta_query(question):
        category_list = ", ".join(CATEGORIES.values())
        answer = (
            f"Calamba City offers services across these categories: "
            f"{category_list}. Ask me about a specific one and I can give you details!"
        )
        return {
            "question": question,
            "predicted_intent": "META",
            "intent_label": "Meta",
            "intent_confidence": 1.0,
            "retrieved_chunks": [],
            "answer": answer,
            "forms": [],
            "history": history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    if is_forms_availability_query(question):
        others = get_all_forms()
        if others:
            by_office: dict[str, list[dict]] = {}
            for form in others:
                by_office.setdefault(form["office"], []).append(form)
            sections = []
            for office, office_forms in by_office.items():
                lines = "\n".join(f"- {f['name']}" for f in office_forms)
                sections.append(f"**{office}**\n{lines}")
            answer = (
                "Here are the downloadable forms I have on file, grouped by "
                "the office they come from:\n\n" + "\n\n".join(sections)
            )
        else:
            answer = NO_MATCHING_FORMS_ANSWER
        return {
            "question": question,
            "predicted_intent": "FORMS_AVAILABILITY",
            "intent_label": "Downloadable Forms",
            "intent_confidence": 1.0,
            "retrieved_chunks": [],
            "answer": answer,
            "forms": others,
            "history": history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    # when user clicks the service from the category, it skips the contextualize, intent classification, and embedding-based retrieval entirely,
    # and just fetch every chunk of that document directly
    if doc_id_hint is not None:
        retrieved = get_chunks_by_doc_id(doc_id_hint)
        log(f"[retrieval] doc_id_hint={doc_id_hint!r} -> "
            f"{[c.get('chunk_id') for c in retrieved]}")
        if retrieved:
            top_office = next((c.get("office") for c in retrieved if c.get("office")), None)
            retrieved = _attach_office_contact(retrieved, top_office)
            answer = generate_answer(question, retrieved, history=history)
            topic = retrieved[0]["title"]
            log(f"[topic] stored for next turn: {topic!r}")
            # forms are shown per office
            forms = get_forms_for_office(top_office) if not is_refusal(answer) else []
            category_label = CATEGORIES.get(category_hint, category_hint) if category_hint else topic
            return {
                "question": question,
                "predicted_intent": category_hint,
                "intent_label": category_label,
                "intent_confidence": 1.0,
                "retrieved_chunks": retrieved,
                "answer": answer,
                "forms": forms,
                "history": history + [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer, "topic": topic},
                ],
            }
        # if ever doc_id_hint somehow matches nothing (stale id, etc.), fall through to the normal pipeline below rather than dead-ending
        log(f"[retrieval] doc_id_hint={doc_id_hint!r} found no chunks, falling back to normal retrieval")


    if category_hint is not None:
        search_query = question
    else:
        search_query = contextualize_question(question, history)

    # intent classification
    intent_result = predict_intent(search_query)
    if category_hint is not None:
        category_filter = category_hint
    else:
        category_filter = intent_result["intent"] if intent_result["confident"] else None

    # retrieval
    if is_enumeration_query(search_query) and category_filter is not None:
        retrieved = retrieve_all_in_category(category_filter)
        titles = sorted(set(c["title"] for c in retrieved if c.get("title")))
        category_label = CATEGORIES.get(category_filter, category_filter)
        listing = "\n".join(f"- {t}" for t in titles)
        answer = f"Here are the services under {category_label}:\n\n{listing}"

        forms = [] 

        shown_intent = category_hint if category_hint is not None else intent_result["intent"]
        shown_confidence = 1.0 if category_hint is not None else round(intent_result["confidence"], 3)

        return {
            "question": question,
            "predicted_intent": shown_intent,
            "intent_label": category_label,
            "intent_confidence": shown_confidence,
            "retrieved_chunks": retrieved,
            "answer": answer,
            "forms": forms,
            "history": history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    unfiltered = retrieve(search_query, top_k=TOP_K, category_filter=None)
    if category_filter is not None:
        filtered = retrieve(search_query, top_k=TOP_K, category_filter=category_filter)
        merged: dict[str, dict] = {}
        for chunk in filtered + unfiltered:
            key = chunk.get("chunk_id") or chunk.get("doc_id") or chunk["title"]
            if key not in merged or chunk["similarity"] > merged[key]["similarity"]:
                merged[key] = chunk
        retrieved = sorted(
            merged.values(), key=lambda c: c["similarity"], reverse=True
        )[:TOP_K]
    else:
        retrieved = unfiltered

    log(f"[retrieval] search_query={search_query!r} -> "
        f"{[(c.get('chunk_id'), round(c.get('similarity', 0), 3)) for c in retrieved]}")

    top_office = None
    if retrieved:
        top_doc_id = retrieved[0].get("doc_id")
        if top_doc_id:
            summary_chunk_id = f"{top_doc_id}_summary_chunk0"
            already_have = any(c.get("chunk_id") == summary_chunk_id for c in retrieved)
            if not already_have:
                summary_chunk = get_chunk_by_id(summary_chunk_id)
                if summary_chunk:
                    retrieved = retrieved + [summary_chunk]
                    log(f"[retrieval] attached summary chunk for top doc: {summary_chunk_id!r}")
        top_office = retrieved[0].get("office")
        retrieved = _attach_office_contact(retrieved, top_office)

    # generation
    answer = generate_answer(question, retrieved, history=history)

    topic = retrieved[0]["title"] if retrieved else None
    log(f"[topic] stored for next turn: {topic!r}")

    # attach downloadable forms per office
    show_forms = (
        bool(retrieved)
        and not is_refusal(answer)
        and retrieved[0].get("similarity") is not None
        and retrieved[0]["similarity"] >= FORMS_MIN_SIMILARITY
    )
    forms = get_forms_for_office(top_office) if show_forms else []

    shown_intent = category_hint if category_hint is not None else intent_result["intent"]
    shown_label = (
        CATEGORIES.get(category_hint, category_hint) if category_hint is not None
        else CATEGORIES.get(intent_result["intent"], intent_result["intent"])
    )
    shown_confidence = 1.0 if category_hint is not None else round(intent_result["confidence"], 3)

    return {
        "question": question,
        "predicted_intent": shown_intent,
        "intent_label": shown_label,
        "intent_confidence": shown_confidence,
        "retrieved_chunks": retrieved,
        "answer": answer,
        "forms": forms,
        "history": history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer, "topic": topic},
        ],
    }