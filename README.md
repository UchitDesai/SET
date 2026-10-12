# SET

# Multilingual Multimodal Phishing Detection API

A cybersecurity project that detects phishing attempts using multilingual text, URLs, email metadata, and uploaded files. The system combines language detection, modality analysis, semantic reasoning, and adaptive risk fusion to generate a phishing risk score and explainable verdict.

## Overview

This project is designed to assess suspicious messages and content across multiple modalities, including:

- Raw text input
- URL analysis
- Email subject and sender metadata
- Uploaded files such as screenshots, documents, and scanned content
- Multilingual detection across major global scripts

It uses a FastAPI backend and combines several detection stages:

- Language identification for multilingual content
- Feature extraction and modality processing
- Reasoning graph generation
- Cross-modal reasoning and category mapping
- Adaptive fusion for final phishing risk scoring
- Session-based memory and semantic retrieval

## Key Features

- Multilingual phishing detection
- Support for Latin and non-Latin scripts
- File-based analysis for uploaded suspicious content
- Cross-modal reasoning between URL, text, and file signals
- Explainable verdict with reasoning details
- Session-based memory for retrieval and historical context
- FastAPI REST API for integration into security platforms

## Tech Stack

- Python
- FastAPI
- Uvicorn
- FAISS
- Sentence Transformers
- langdetect
- Pillow
- Scikit-learn
- PyTorch
- SQLite for session memory

## Project Structure

```text
phishing_project/
├── app/
│   ├── api/
│   │   └── endpoints.py
│   ├── services/
│   │   ├── adaptive_fusion_engine.py
│   │   ├── cross_modal_reasoner.py
│   │   ├── language_detector.py
│   │   ├── llm_interface.py
│   │   ├── memorymanager.py
│   │   ├── ml_model.py
│   │   ├── modality_processor.py
│   │   ├── phishing_extractor.py
│   │   ├── phishing_features.py
│   │   ├── reasoning_graph.py
│   │   ├── risk_engine.py
│   │   ├── semantic_signal_mapper.py
│   │   └── vector_store.py
│   └── utils/
├── data/
│   └── Raw/
├── memory_store/
├── uploads/
├── vector_store/
├── main.py
├── requirements.txt
├── test_run.py
└── README.md
```

## Installation

1. Clone the repository:
   git clone https://github.com/UchitDesai/SET.git

2. Change into the project directory:
   cd SET

3. Create a virtual environment:
   python -m venv venv

4. Activate the environment:
   - Windows:
     venv\Scripts\activate
   - macOS/Linux:
     source venv/bin/activate

5. Install dependencies:
   pip install -r requirements.txt

## Running the Application

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

Then open the API documentation at:

- http://127.0.0.1:8000/docs
- http://127.0.0.1:8000/redoc

## API Endpoints

### 1. Start a session

POST /start_session

Returns a new session ID for tracking analysis history.

### 2. Analyze phishing content

POST /analyze_phishing

Accepts:

- session_id
- raw_text
- url
- email_subject
- email_sender
- file (optional upload)

Returns:

- detected language
- script family
- risk score
- final verdict
- reasoning graph
- explanation
- debugging context

## Example Usage

### Using curl

```bash
curl -X POST "http://127.0.0.1:8000/start_session"
```

Then send input to the phishing analysis endpoint with the returned session ID.

## Model and Logic Notes

This project combines heuristic phishing indicators with semantic features and reasoning-based scoring. The architecture is designed to support explainability rather than black-box classification alone.

Important components include:

- `language_detector.py` for multilingual language/script detection
- `modality_processor.py` for merging content modalities
- `reasoning_graph.py` for structured phishing reasoning
- `cross_modal_reasoner.py` for inter-modal signal evaluation
- `adaptive_fusion_engine.py` for final verdict generation
- `vector_store.py` for FAISS-based session memory and retrieval

## Use Cases

- Security analysis of suspicious email content
- Detection of phishing text in multiple languages
- Risk classification for marketing or fraud monitoring systems
- Explainable AI for cybersecurity workflows

## License

This project is distributed under the MIT License. See the LICENSE file if included in your repository.

## Author

Developed as an AI-powered phishing detection and reasoning project for cybersecurity analysis.

## GitHub Upload Notes

After creating your repository on GitHub, initialize git and push the project with:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/UchitDesai/SET.git
git push -u origin main
```

## Disclaimer

This project is intended for educational and research purposes in cybersecurity. It should be used responsibly and in accordance with applicable laws and institutional policies.
