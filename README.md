# CAL-E: A RAG-Powered Calamba Assistant for City Information and Services

CAL-E is an AI-based chatbot designed to simplify public access to Calamba City government information. Instead of forcing users to navigate through multiple layers of a traditional website, CAL-E allows users to ask natural language questions (e.g., "What are the requirements for a business permit?") and receive immediate, grounded responses.

## 🌟 Key Features

* **Retrieval-Augmented Generation (RAG):** Documents are chunked, embedded using `all-MiniLM-L6-v2`, and stored in ChromaDB for fast, relevant retrieval.


* **Fully Local Deployment:** The system generates responses using the `Qwen2.5-7B-Instruct` language model run locally via Ollama. This ensures user privacy and eliminates cloud usage costs.


* **Hybrid NLP Pipeline:** Combines LLM generation with a traditional NLP/ML intent classification layer.


* **Automated Form Lookup:** Directly surfaces relevant government forms (like the Barangay Jurisdiction Request Form) to the user based on intent.



## 📂 Repository Structure

The project is structured into modular components separating the frontend, backend, and data processing layers:

* **`backend/`**

* `app.py`: The main application server routing requests.


* `build_index.py`: Script to process documents and build the ChromaDB vector index.


* `rag_pipeline.py` & `generate.py`: Core components handling retrieval and LLM generation.


* `intent_classifier.py` & `preprocess.py`: Handles the NLP intent classification layer.


* `forms_lookup.py`: Maps intents to specific local government PDF forms.


* `evaluate.py`: Contains standard evaluation metrics for the RAG and classification pipeline.




* **`frontend/`**

* Contains the web interface, including `templates/index.html` and static assets (`script.js`, `style.css`).




* **`data/`**

* `documents.json`: Chunked raw text data from the Calamba City website.


* `intent_training_data.csv`: Training data for the intent classification model.


* `forms/`: Stores downloadable PDF assets, such as `barangay_jurisdiction_request_form.pdf`.


* `review/`: Contains manual review reports and schema documentation.





## 🚀 Getting Started

### Prerequisites

1. **Python 3.8+**
2. **Ollama:** Must be installed and running locally to serve the LLM.

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/calamba-chatbot.git
cd calamba-chatbot

```


2. **Install Python dependencies:**
Ensure you are in a virtual environment, then install the required packages:
```bash
pip install -r requirements.txt

```


3. **Pull the local LLM via Ollama:**
```bash
ollama run qwen2.5:7b-instruct

```



### Running the Application

1. **Build the Vector Index:**
Before starting the chat, you need to embed `documents.json` into ChromaDB.


```bash
python backend/build_index.py

```


2. **Start the Backend Server:**
```bash
python backend/app.py

```


3. **Access the Interface:**
Open your browser and navigate to `http://localhost:5000` (or whichever port `app.py` specifies) to interact with CAL-E.

---
