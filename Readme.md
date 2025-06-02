# Biomedical AI Assistant (RAG-powered)

This project is an advanced **Retrieval-Augmented Generation (RAG)** application designed to answer **biomedical questions** using AI and domain-specific document search. It leverages **BioBERT embeddings**, a **local quantized Mistral-based LLM**, and **FastAPI** for a lightweight and deployable medical chatbot experience.

---

## Features

- **Local LLM** using quantized `openhermes-2-mistral-7b.Q4_K_M.gguf`
- **BioBERT sentence embeddings** for domain-specific semantic search
- **FastAPI web app** with an intuitive UI (Jinja2 templates)
- **Qdrant vector store** for fast chunk retrieval from medical PDFs
- Fully offline, efficient, and tailored for biomedical queries

---

## Project Structure

```bash
.
├── rag.py                    # FastAPI app to run RAG chatbot
├── ingest.py                 # PDF loading and vector index creation
├── retriever.py              # Script to test similarity search
├── templates/index.html      # Web UI for question submission
├── static/                   # Static assets (CSS, JS if needed)
├── Data/                     # Folder with medical PDFs
├── requirements.txt          # Python dependencies
└── openhermes-2-mistral-7b.Q4_K_M.gguf # Local quantized LLM model (can be found on [HuggingFace](https://huggingface.co/TheBloke/OpenHermes-2-Mistral-7B-GGUF/tree/main))
```

## Tech Stack

| Component         | Tool / Model                                              |
| ----------------- | --------------------------------------------------------- |
| Language Model | `openhermes-2-mistral-7b.Q4_K_M.gguf` via CTransformers   |
| Embeddings     | `pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb` |
| Vector Store   | Qdrant (self-hosted)                                      |
| Backend        | FastAPI                                                   |
| UI             | Jinja2 Templates (HTML)                                   |


## Installation
1. Clone the Repository
```bash
git clone <>
```
2. Install Dependencies
```bash
pip install -r requirements.txt
```
3. Download the GGUF Model
Download openhermes-2-mistral-7b.Q4_K_M.gguf (can be found on [HuggingFace](https://huggingface.co/TheBloke/OpenHermes-2-Mistral-7B-GGUF/tree/main)) and place it in the root directory.

## Ingest Medical PDFs
Place your .pdf files into the Data/ folder.

Then run:

```bash
python ingest.py
```

This :
Loads all PDFs

Splits text into chunks

Embeds them using BioBERT

Saves them in Qdrant vector store
