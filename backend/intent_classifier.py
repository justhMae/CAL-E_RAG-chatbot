import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate

from backend.config import INTENT_DATA_PATH, INTENT_MODEL_PATH
from backend.preprocess import clean_text
from backend.embedding_transformer import EmbeddingTransformer


def load_intent_data() -> pd.DataFrame:
    df = pd.read_csv(INTENT_DATA_PATH)
    df = df.dropna(subset=["text"]) # drop rows with missing text
    df["text"] = df["text"].apply(clean_text)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("embed", EmbeddingTransformer()),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])


def load_intent_data() -> pd.DataFrame:
    df = pd.read_csv(INTENT_DATA_PATH)
    df["text"] = df["text"].apply(clean_text)
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("embed", EmbeddingTransformer()),
        ("clf", LogisticRegression(
            max_iter=2000,
            C=1.0, 
            class_weight="balanced",
            solver="lbfgs",
        )),
    ])

def train_and_save(test_size: float = 0.2, random_state: int = 42):
    df = load_intent_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["intent"],
        test_size=test_size, random_state=random_state, stratify=df["intent"],
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    joblib.dump(
        {"pipeline": pipeline, "X_test": X_test, "y_test": y_test,
         "X_train": X_train, "y_train": y_train},
        INTENT_MODEL_PATH,
    )
    train_acc = pipeline.score(X_train, y_train)
    test_acc = pipeline.score(X_test, y_test)
    print(f"Train accuracy: {train_acc:.3f} | Test accuracy: {test_acc:.3f}")
    return pipeline


def load_pipeline() -> Pipeline:
    bundle = joblib.load(INTENT_MODEL_PATH)
    return bundle["pipeline"]


def predict_intent(question: str, confidence_threshold: float = 0.25) -> dict:
    pipeline = load_pipeline()
    cleaned = clean_text(question)
    probs = pipeline.predict_proba([cleaned])[0]
    classes = pipeline.classes_
    best_idx = probs.argmax()
    return {
        "intent": classes[best_idx],
        "confidence": float(probs[best_idx]),
        "confident": bool(probs[best_idx] >= confidence_threshold),
    }


if __name__ == "__main__":
    train_and_save()