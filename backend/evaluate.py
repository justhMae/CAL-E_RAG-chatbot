import argparse
import json
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split

from backend.config import INTENT_MODEL_PATH, EVAL_OUTPUT_DIR
from backend.intent_classifier import load_intent_data, build_pipeline


def evaluate_intent_classifier():
    EVAL_OUTPUT_DIR.mkdir(exist_ok=True)

    df = load_intent_data()
    X, y = df["text"], df["intent"]

    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=True)

    print("\n=== Stratified 5-Fold Cross-Validation ===")
    for metric in scoring:
        train_vals = scores[f"train_{metric}"]
        test_vals = scores[f"test_{metric}"]
        print(f"{metric:20s} | Train: {train_vals.mean():.3f} (±{train_vals.std():.3f}) | Test: {test_vals.mean():.3f} (±{test_vals.std():.3f})")

    pipeline.fit(X, y)
    joblib.dump(
        {"pipeline": pipeline, "X": X, "y": y},
        INTENT_MODEL_PATH,
    )
    print(f"\nFinal model saved to {INTENT_MODEL_PATH}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    report = classification_report(y_test, y_pred, zero_division=0)

    print(f"\nAccuracy:  {acc:.3f}")
    print(f"Precision (macro): {precision:.3f}")
    print(f"Recall (macro):    {recall:.3f}")
    print(f"F1-score (macro):  {f1:.3f}\n")
    print(report)

    with open(EVAL_OUTPUT_DIR / "classification_report.txt", "w") as f:
        f.write(f"Accuracy:  {acc:.3f}\n")
        f.write(f"Precision (macro): {precision:.3f}\n")
        f.write(f"Recall (macro):    {recall:.3f}\n")
        f.write(f"F1-score (macro):  {f1:.3f}\n\n")
        f.write(report)

    # confusion matrix plot
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(9, 9))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, xticks_rotation=90, cmap="Greens", colorbar=False)
    plt.title("Intent Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig(EVAL_OUTPUT_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved confusion matrix to {EVAL_OUTPUT_DIR / 'confusion_matrix.png'}")

    # error analysis
    errors = []
    for text, true_label, pred_label in zip(X_test, y_test, y_pred):
        if true_label != pred_label:
            errors.append({"text": text, "true": true_label, "predicted": pred_label})

    with open(EVAL_OUTPUT_DIR / "intent_error_analysis.json", "w") as f:
        json.dump(errors, f, indent=2)
    print(f"{len(errors)} misclassified example(s) saved to intent_error_analysis.json")


SAMPLE_QUESTIONS = [
    "What are the requirements for a business permit?",
    "Where can I get a free anti-rabies vaccine for my dog?",
    "How do I apply for a city scholarship?",
    "What should I do during a strong earthquake?",
    "How do I pay my real property tax?",
    "Can you tell me a joke?",
]


def run_sample_conversations():
    from backend.rag_pipeline import answer_question

    EVAL_OUTPUT_DIR.mkdir(exist_ok=True)
    lines = ["# Sample Conversations\n"]
    for q in SAMPLE_QUESTIONS:
        result = answer_question(q)
        lines.append(f"**User:** {q}\n")
        lines.append(
            f"**Predicted intent:** {result['intent_label']} "
            f"({result['intent_confidence']*100:.0f}% confidence)\n"
        )
        lines.append(f"**Bot:** {result['answer']}\n")
        lines.append("---\n")
        print(f"Q: {q}\nA: {result['answer']}\n")

    with open(EVAL_OUTPUT_DIR / "sample_conversations.md", "w") as f:
        f.write("\n".join(lines))
    print(f"Saved sample conversations to {EVAL_OUTPUT_DIR / 'sample_conversations.md'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag", action="store_true", help="Also run sample RAG conversations")
    args = parser.parse_args()

    evaluate_intent_classifier()

    if args.rag:
        run_sample_conversations()
    else:
        print("\nSkipped RAG sample conversations")