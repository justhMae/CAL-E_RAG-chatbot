# intent classifier: TF-IDF (word 1-2 grams) + LogisticRegression
# runtime only needs scikit-learn, numpy and joblib (no sentence-transformers / torch)
# re-train and re-save backend/intent_model.joblib with:  python -m backend.intent_classifier

import csv

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from backend.config import INTENT_DATA_PATH, INTENT_MODEL_PATH
from backend.preprocess import clean_text

DEFAULT_MODE = "tfidf"


def load_intent_data() -> tuple[list[str], list[str]]:
    # returns (texts, intents); rows with missing text are dropped
    texts, intents = [], []
    with open(INTENT_DATA_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text = clean_text(row.get("text"))
            intent = (row.get("intent") or "").strip()
            if not text or not intent:
                continue
            texts.append(text)
            intents.append(intent)
    return texts, intents


def _tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)


def build_pipeline(mode: str = DEFAULT_MODE) -> Pipeline:
    if mode != "tfidf":
        raise ValueError(
            f"unknown mode: {mode!r}. Only 'tfidf' is supported; the embedding-based modes were "
            "removed so the deployed classifier has no sentence-transformers/torch dependency."
        )

    return Pipeline([
        ("features", _tfidf()),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=10.0,  # 5-fold CV on the TF-IDF-only features: C=1 -> 0.76, C=10 -> 0.81 accuracy
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def save_pipeline(pipeline: Pipeline, mode: str = DEFAULT_MODE) -> None:
    # store plain Python objects only, so loading never needs pandas
    joblib.dump({"pipeline": pipeline, "mode": mode}, INTENT_MODEL_PATH, compress=3)


def train_and_save(test_size: float = 0.2, random_state: int = 42, mode: str = DEFAULT_MODE):
    texts, intents = load_intent_data()
    X_train, X_test, y_train, y_test = train_test_split(
        texts, intents,
        test_size=test_size, random_state=random_state, stratify=intents,
    )

    pipeline = build_pipeline(mode)
    pipeline.fit(X_train, y_train)
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"[{mode}] Train accuracy: {train_acc:.3f} | Test accuracy: {test_acc:.3f}")

    # the deployed model is re-fit on all data after the held-out check
    pipeline = build_pipeline(mode)
    pipeline.fit(texts, intents)
    save_pipeline(pipeline, mode)
    print(f"Saved {mode} model trained on {len(texts)} examples to {INTENT_MODEL_PATH}")
    return pipeline


_pipeline = None


def load_pipeline() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = joblib.load(INTENT_MODEL_PATH)["pipeline"]
    return _pipeline


def predict_intent(question: str, confidence_threshold: float = 0.25) -> dict:
    pipeline = load_pipeline()
    cleaned = clean_text(question)
    probs = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    best_idx = probs.argmax()
    return {
        "intent": str(classes[best_idx]),
        "confidence": float(probs[best_idx]),
        "confident": bool(probs[best_idx] >= confidence_threshold),
    }


if __name__ == "__main__":
    train_and_save()
