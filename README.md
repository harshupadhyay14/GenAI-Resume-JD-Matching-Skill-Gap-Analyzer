HEAD
# 🤖 GenAI Resume–JD Matcher

> AI-powered resume analysis, skill gap detection, cover letter generation & interview prep — built with FastAPI + Groq LLaMA 3.3 + FAISS.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-Semantic_Search-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Match Score** | Semantic similarity score between resume & JD using FAISS + Sentence Transformers |
| 🎯 **Skill Gap Analysis** | Extracts skills across 9 categories, identifies matched & missing skills |
| 📈 **Category Scores** | Scores across Technical, GenAI/LLM, Data, Cloud, and Soft Skills |
| ✉️ **Cover Letter Generator** | One-click tailored cover letter using Groq LLaMA 3.3 70B |
| ✏️ **Bullet Rewriter** | Rewrites weak resume bullets into 3 ATS-optimized versions |
| 🎯 **Interview Prep** | 10 personalized interview questions based on your resume + JD gaps |

---

## 🏗️ Architecture

```
resume_jd_matcher/
├── main.py                  ← FastAPI app, all routes
├── app/
│   ├── parser.py            ← PDF/DOCX text extraction (PyMuPDF)
│   ├── embedder.py          ← FAISS + Sentence Transformers similarity & skill extraction
│   ├── analyzer.py          ← Groq LLaMA 3.3 — cover letter, bullet rewrite, interview prep
│   └── models.py            ← Pydantic schemas
├── templates/
│   └── index.html           ← Dark UI — radar chart, tabs, skill tags
├── Dockerfile
├── render.yaml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/YOUR_USERNAME/genai-resume-jd-matcher.git
cd genai-resume-jd-matcher
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux
pip install -r requirements.txt
```

### 2. Add API Key
```bash
# Create .env file
echo GROQ_API_KEY=gsk_your_key_here > .env
```
Get a free Groq API key at [console.groq.com](https://console.groq.com)

### 3. Run
```bash
uvicorn main:app --reload --port 8000
```

Open [http://localhost:8000](http://localhost:8000)

---

## 🌐 Deploy on Render

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New → Web Service → Connect repo
3. Add environment variable: `GROQ_API_KEY` = your key
4. Deploy — live URL in ~3 minutes

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.11
- **AI/LLM**: Groq API (LLaMA 3.3 70B)
- **Embeddings**: Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Search**: FAISS
- **File Parsing**: PyMuPDF, python-docx
- **Frontend**: Vanilla JS, Chart.js, dark UI
- **Deployment**: Render, Docker

---

## 📄 License

MIT © Harsh Upadhyay

# GenAI-Resume-JD-Matching-Skill-Gap-Analyzer

