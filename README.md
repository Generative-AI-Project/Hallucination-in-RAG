# Self-Healing RAG: A Multi-Agent Framework for Hallucination Detection & Correction

A multi-agent Retrieval-Augmented Generation (RAG) pipeline that autonomously detects and corrects LLM hallucinations. This system goes beyond passive evaluation by integrating FAISS-based dense retrieval, Knowledge Graph (KG) triplet alignment, NLI-based hallucination detection, and an autonomous correction routing loop.

Developed as a research initiative and submitted to the IEEE Cyber-AI 2026 conference. 

## 🚀 System Architecture

The pipeline executes four sequential stages with conditional branching, orchestrated entirely as a `LangGraph` StateGraph.

1. **Retrieval Agent:** Conducts top-k similarity search over dual FAISS indexes (SQuAD v1.1 and WikiBio).
2. **Generator Agent:** Generates an initial response using `llama-3.1-8b-instant`.
3. **Detector Agent:** Extracts Subject-Predicate-Object (SPO) triplets, constructs directed Knowledge Graphs, and scores claims via cosine similarity and `cross-encoder/nli-deberta-v3-small` NLI scoring.
4. **Corrector Agent:** Triggered when the hallucination score is ≥ 0.50. Rewrites answers strictly using retrieved context at Temperature = 0.0.

## 📊 Key Results
* **Retrieval Quality:** Improved relevant chunk recall from **2.0 to 3.5** (in top-5 results) compared to baseline embedding models.
* **Detection Performance:** Achieved robust sentence-level hallucination detection on the WikiBio GPT-4o benchmark using hybrid KG+NLI scoring.
* **Self-Healing:** Successfully prevented confident fabrications via strict context-grounded fallback clauses.

## 🛠️ Technology Stack
* **Orchestration:** LangGraph, LangChain
* **Retrieval & Vector Store:** FAISS, `multi-qa-MiniLM-L6-cos-v1` embeddings
* **Language Models:** `llama-3.1-8b-instant` (via Groq API)
* **Knowledge Graphs & NLI:** `networkx`, `cross-encoder/nli-deberta-v3-small`
* **UI/Deployment:** Streamlit, ngrok

## 👥 Contributors & Work Distribution

This project was developed by a 3-person engineering team at PES University.

* **Mahima Shettigar  — Retrieval & Generation Modules**
  * Engineered the dual-knowledge-base dense retrieval pipeline and batched FAISS vectorstore construction.
  * Optimized document chunking and selected MiniLM embeddings, directly increasing relevant chunk recall.
  * Built the context-grounded generator agent within the LangGraph state machine.
* **V Sreenidhe — Hallucination Detection Module**
  * Built the SPO triplet extraction logic, Knowledge Graph construction, and cosine edge alignment.
  * Implemented the DeBERTa sentence-level NLI scoring and the hybrid thresholding algorithm.
* **Vasavi Bolar — Orchestration & Correction Module**
  * Architected the LangGraph StateGraph, conditional routing logic, and correction agent prompts.
  * Developed the Streamlit front-end with per-claim KG visualization.

## ⚙️ Local Setup & Installation

1. **Clone the repository:**
```bash
   git clone [https://github.com/Generative-AI-Project/Hallucination-in-RAG.git](https://github.com/Generative-AI-Project/Hallucination-in-RAG.git)
   cd Hallucination-in-RAG
   ```

2. **Install dependencies:**
```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
```env
   GROQ_API_KEY=your_groq_api_key
   ```

4. **Run the Streamlit Application:**
```bash
   streamlit run app.py
   ```
