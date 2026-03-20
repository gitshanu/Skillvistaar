# SkillVistaar-CRAG 🎓
### AI Career Coach for Indian Students — Powered by Corrective RAG

A production-ready **Corrective Retrieval-Augmented Generation (CRAG)** system 
built with LangGraph that helps Indian students navigate NEP 2020 policies, 
AI literacy, and job market trends — without hallucinating.



## The Problem it Solves

Generic chatbots give **wrong or outdated advice** to students asking about:
- NEP 2020 / NCF-SE 2023 policy details
- Career paths in AI and tech
- Current job market skills (India Skills Report 2026)

**SkillVistaar fixes this** using CRAG — it self-corrects its retrieval before 
generating any answer, and falls back to live web search when local docs are insufficient.



## How CRAG Works (the pipeline)
```
User Question
      ↓
  RETRIEVE → top 2 chunks from FAISS
      ↓
   GRADE  → LLM checks: are these chunks actually relevant?
      ↓
  YES ──────────────────────→ GENERATE → Answer
  NO  → REWRITE → RETRIEVE → GRADE
                                  ↓
                    still NO → WEB SEARCH → GENERATE → Answer
```



## Tech Stack (100% Free & Local)
```
| Component | Tool |
|---|---|
| Orchestration | LangGraph |
| LLM | Ollama — llama3.2 (local) |
| Embeddings | all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS (in-memory) |
| Web Search | Tavily API (free tier) |
| Frontend | Streamlit |

```
## Knowledge Base

- 📄 NEP 2020 — National Education Policy
- 📄 NCF-SE 2023 — National Curriculum Framework
- 📄 India Skills Report 2026
- 🌐 Wikipedia (AI, Skill India, NEP — auto-loaded)
- 🔍 Tavily live web search (real-time fallback)



## Project Structure
```
skillvistaar-crag/
├── src/
│   ├── ingestion.py     # Downloads + chunks PDFs
│   ├── embeddings.py    # FAISS vectorstore builder
│   ├── retriever.py     # Semantic search
│   ├── grader.py        # CRAG relevance grader
│   ├── rewriter.py      # Query rewriter
│   ├── generator.py     # Answer generator
│   └── graph.py         # LangGraph CRAG state machine
├── app.py               # Streamlit UI
├── config.py            # Central settings
├── data/                # PDFs downloaded here (auto)
├── vectorstore/         # FAISS index saved here (auto)
└── requirements.txt
```



## Setup & Run

### 1. Clone the repo

git clone https://github.com/YOUR_USERNAME/Skillvistaar.git
cd Skillvistaar


### 2. Create virtual environment

python -m venv .venv
.venv\Scripts\activate       # Windows
source .venv/bin/activate    # Mac/Linux


### 3. Install dependencies

pip install -r requirements.txt


### 4. Install Ollama + pull model
Download from https://ollama.com then:

ollama pull llama3.2
ollama serve


### 5. Set up environment variables
Create a .env file:

TAVILY_API_KEY=your_tavily_key_here

Get free key at https://app.tavily.com

### 6. Build the vectorstore (first time only)

python src/embeddings.py

This downloads PDFs and builds the FAISS index (~2-3 minutes).

### 7. Run the app

streamlit run app.py

Open http://localhost:8501 in your browser.


## Sample Questions to Try

- *"What does NEP 2020 say about vocational education?"*
- *"What are the top AI skills required for jobs in India 2026?"*
- *"What is the 5+3+3+4 structure in NEP 2020?"*
- *"What is the latest news in AI hiring this week?"* ← triggers web search


