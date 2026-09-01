# Calamba City Info Chatbot — RAG + Intent Classification

A domain chatbot that answers questions about Calamba City government
services (permits, taxes, health, housing, etc.) using Retrieval-Augmented
Generation, plus a scikit-learn intent classifier for the ML-evaluation
part of the rubric. Fully local — no API keys, no cloud costs.

```
User Question
   -> Text Preprocessing
   -> Intent Classification (LogisticRegression on multilingual-MiniLM embeddings)
   -> paraphrase-multilingual-MiniLM-L12-v2 Embedding (sentence-transformers)
   -> ChromaDB similarity search
   -> Qwen2.5-7B-Instruct (via Ollama) generates the final answer
```

**Chunking technique:** one chunk per service, not fixed-size sliding
windows. Each entry in `data/documents.json` is a structured record
(office, requirements table, process-flow table, fees) — splitting that
by character count would separate a fee from the step that produced it.
`backend/preprocess.py` flattens each service into one self-contained
chunk (falling back to sliding-window splitting only if a single service
is unusually long, >2200 characters).

---

## Run it in Google Colab (recommended if your laptop can't run a 7B model)

Open `colab/run_in_colab.ipynb` in Google Colab, set the runtime to
**GPU (T4)** (Runtime → Change runtime type), and run the cells top to
bottom. It handles everything: uploading the project, installing Ollama,
pulling Qwen2.5-7B-Instruct, installing dependencies, building the index,
training the classifier, running evaluation, and launching the web app
through an ngrok tunnel so you get a public link to demo from.

You'll need a free ngrok account (https://dashboard.ngrok.com/signup) for
the last step — the notebook tells you exactly where to paste your
authtoken.

---

## Run it locally instead

### 1. Install Ollama and pull the model

Get Ollama from **https://ollama.com/download**, then:

```bash
ollama pull qwen2.5:7b-instruct
ollama serve   # leave this running in its own terminal
```

Qwen2.5-7B-Instruct needs a reasonably capable machine (ideally 16GB+ RAM,
faster with a GPU). If your laptop struggles, use the Colab notebook above
instead — that's exactly what it's for.

### 2. Python setup

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

No `.env` / API key needed — `sentence-transformers` downloads
all-MiniLM-L6-v2 automatically the first time it runs (~80MB, cached
after that), and Ollama runs entirely on your machine.

### 3. Build the vector index

```bash
python -m backend.build_index
```

Embeds every chunk in `data/documents.json` with MiniLM and stores it in
a local ChromaDB folder (`chroma_store/`). Re-run this any time you edit
`data/documents.json`.

### 4. Train the intent classifier

```bash
python -m backend.intent_classifier
```

Trains on `data/intent_training_data.csv`, saves to
`backend/intent_model.joblib`, and prints train/test accuracy.

### 5. Run the chatbot

```bash
python -m backend.app
```

Open **http://localhost:5000**.

### 6. Generate evaluation results (for the report/presentation)

```bash
python -m backend.evaluate            # intent metrics only
python -m backend.evaluate --rag       # + sample RAG conversations (needs Ollama running)
```

Produces, in `evaluation/`:
- `classification_report.txt` — Accuracy, Precision, Recall, F1-score (macro)
- `confusion_matrix.png`
- `intent_error_analysis.json` — misclassified examples
- `sample_conversations.md` — example Q&A pairs, including one
  deliberately out-of-domain question for your error analysis

---

## Data status

- **`data/documents.json`** — 192 real services scraped from the Calamba
  City website, reviewed and cleaned. See `data/review/README.md` for the
  history of what was auto-parsed vs. manually fixed, and what (if
  anything) still needs a second look.
- **Downloadable forms** — the `forms` feature (attach a downloadable
  file to a service, see `data/SCHEMA.md`) is wired up end-to-end, but
  the actual form files haven't been downloaded from the website yet.
  Only one placeholder example (`doc_tax_004` — Barangay Jurisdiction) has
  a file in `data/forms/`. Add more by dropping files in `data/forms/`
  and adding a `"forms"` entry to the relevant document.
- **`data/intent_training_data.csv`** — still the original placeholder
  set (~10 examples per category). Worth expanding with real phrasings
  now that the category distribution has shifted (Health, Education, and
  Registrations are much bigger than originally assumed) — see the
  category breakdown in `data/review/README.md`.

---

## Project structure

```
calamba-chatbot/
├── colab/
│   └── run_in_colab.ipynb          # run everything on Colab's free GPU
├── data/
│   ├── documents.json              # RAG knowledge base (192 real services)
│   ├── intent_training_data.csv    # labeled questions for the classifier
│   ├── forms/                      # downloadable form files
│   ├── SCHEMA.md                   # documents.json field reference
├── backend/
│   ├── config.py                   # paths, model names, categories
│   ├── preprocess.py               # cleaning + chunking
│   ├── embeddings.py               # MiniLM embeddings + ChromaDB
│   ├── generate.py                 # Qwen2.5-7B-Instruct via Ollama
│   ├── intent_classifier.py        # TF-IDF + Logistic Regression
│   ├── forms_lookup.py             # doc_id -> downloadable forms
│   ├── build_index.py              # script: build the vector index
│   ├── rag_pipeline.py             # ties it all together
│   ├── evaluate.py                 # metrics, confusion matrix, sample convos
│   └── app.py                      # Flask app (API + serves frontend + form downloads)
├── frontend/
│   ├── templates/index.html
│   └── static/{style.css, script.js}
├── evaluation/                     # generated results land here
└── requirements.txt
```
