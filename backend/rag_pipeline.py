import re
from backend.preprocess import clean_text, is_greeting, correct_typos, build_domain_vocabulary
from backend.intent_classifier import predict_intent
from backend.embeddings import retrieve, retrieve_all_in_category, get_chroma_collection
from backend.generate import generate_answer
from backend.forms_lookup import get_forms_for_doc_ids, get_other_forms
from backend.config import TOP_K, CATEGORIES, FORMS_MIN_SIMILARITY

GREETING_RESPONSE = (
    "Hello! I'm the virtual assistant for the City Government of Calamba. "
    "I can help with questions about certificates & permits, business & trade, "
    "health, education, tax, housing, disaster preparedness, and other city "
    "services. What would you like to know?"
)

ENUMERATION_PATTERNS = re.compile(
    r"\b(all|list|every|what are the)\b.*\b(services?|requirements?|programs?|documents?)\b",
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
    "All of our downloadable forms are already linked to their matching "
    "service above -- there aren't any extra standalone files right now."
)

REFUSAL_PATTERN = re.compile(
    r"\b(not sure|does not contain|doesn'?t contain|no information (about|on|regarding))\b",
    re.IGNORECASE,
)


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_PATTERN.search(answer))

_vocab = None


def is_enumeration_query(question: str) -> bool:
    return bool(ENUMERATION_PATTERNS.search(question))


def is_meta_query(question: str) -> bool:
    return bool(META_QUERY_PATTERN.search(question))


def is_forms_availability_query(question: str) -> bool:
    return bool(FORMS_AVAILABILITY_PATTERN.search(question))


def answer_question(question: str, history: list[dict] | None = None) -> dict:
    global _vocab
    question = clean_text(question)
    history = history or []

    if _vocab is None:
        _vocab = build_domain_vocabulary(get_chroma_collection())
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
        others = get_other_forms()
        if others:
            by_office: dict[str, list[dict]] = {}
            for form in others:
                by_office.setdefault(form["office"], []).append(form)
            sections = []
            for office, office_forms in by_office.items():
                lines = "\n".join(f"- {f['name']}" for f in office_forms)
                sections.append(f"**{office}**\n{lines}")
            answer = (
                "Here are other downloadable forms that aren't tied to a "
                "specific service above:\n\n" + "\n\n".join(sections)
            )
        else:
            answer = NO_MATCHING_FORMS_ANSWER
        return {
            "question": question,
            "predicted_intent": "FORMS_AVAILABILITY",
            "intent_label": "Other Downloadable Forms",
            "intent_confidence": 1.0,
            "retrieved_chunks": [],
            "answer": answer,
            "forms": others,
            "history": history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    # 1. Intent classification
    intent_result = predict_intent(question)
    category_filter = intent_result["intent"] if intent_result["confident"] else None

    # 2. Retrieval
    if is_enumeration_query(question) and category_filter is not None:
        retrieved = retrieve_all_in_category(category_filter)
        titles = sorted(set(c["title"] for c in retrieved if c.get("title")))
        category_label = CATEGORIES.get(category_filter, category_filter)
        listing = "\n".join(f"- {t}" for t in titles)
        answer = f"Here are the services under {category_label}:\n\n{listing}"

        doc_ids = [c["doc_id"] for c in retrieved if c.get("doc_id")]
        forms = get_forms_for_doc_ids(doc_ids)

        return {
            "question": question,
            "predicted_intent": intent_result["intent"],
            "intent_label": category_label,
            "intent_confidence": round(intent_result["confidence"], 3),
            "retrieved_chunks": retrieved,
            "answer": answer,
            "forms": forms,
            "history": history + [
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer},
            ],
        }

    unfiltered = retrieve(question, top_k=TOP_K, category_filter=None)
    if category_filter is not None:
        filtered = retrieve(question, top_k=TOP_K, category_filter=category_filter)
        merged: dict[str, dict] = {}
        for chunk in filtered + unfiltered:
            key = chunk.get("doc_id") or chunk["title"]
            if key not in merged or chunk["similarity"] > merged[key]["similarity"]:
                merged[key] = chunk
        retrieved = sorted(
            merged.values(), key=lambda c: c["similarity"], reverse=True
        )[:TOP_K]
    else:
        retrieved = unfiltered

    # 3. Generation
    answer = generate_answer(question, retrieved, history=history)

    # 4. Attach downloadable forms
    doc_ids = [
        c["doc_id"] for c in retrieved
        if c.get("doc_id") and c.get("similarity") is not None
        and c["similarity"] >= FORMS_MIN_SIMILARITY
    ]
    if is_refusal(answer):
        doc_ids = []
    forms = get_forms_for_doc_ids(doc_ids)

    return {
        "question": question,
        "predicted_intent": intent_result["intent"],
        "intent_label": CATEGORIES.get(intent_result["intent"], intent_result["intent"]),
        "intent_confidence": round(intent_result["confidence"], 3),
        "retrieved_chunks": retrieved,
        "answer": answer,
        "forms": forms,
        "history": history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    }