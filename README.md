# 🤖 GenAI Resume–JD Matching & Skill Gap Analyzer

> AI-powered resume analysis, semantic skill gap detection, cover letter generation & interview prep — built with FastAPI + Groq LLaMA 3.3 70B.

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?style=flat-square&logo=fastapi)
![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-TF--IDF-red?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

🔗 **Live Demo:** [genai-resume-jd-matching-skill-gap.onrender.com](https://genai-resume-jd-matching-skill-gap.onrender.com)

---

## 📸 Screenshots

<img width="960" alt="Match Score" src="https://github.com/user-attachments/assets/93fb9402-7857-4d2b-a9fa-4efd81baf10c" />

<img width="960" alt="Skill Gap Analysis" src="https://github.com/user-attachments/assets/411417b4-ac86-49cb-a200-92aa1cc29857" />

<img width="960" alt="Radar Chart" src="https://github.com/user-attachments/assets/b0615f55-b121-4b2d-9f62-c34d4b47510d" />

<img width="960" alt="Cover Letter Generator" src="https://github.com/user-attachments/assets/4893595d-ba98-42d6-aa46-8466d1020c52" />

<img width="960" alt="Bullet Rewriter" src="https://github.com/user-attachments/assets/6024a1e8-8751-4d66-8e1a-9fab59b6a10c" />

<img width="960" alt="Interview Prep" src="https://github.com/user-attachments/assets/d3b4c0aa-f9c0-4e84-b689-190382589692" />

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Semantic Match Score** | Groq LLaMA 3.3 70B rates resume–JD fit (0–100%) with deep contextual understanding |
| 🎯 **Skill Gap Analysis** | Extracts and compares skills across 9 categories — identifies matched & missing skills |
| 📈 **Category Scores** | Radar chart scoring across Technical, Experience, Education, Projects, and Soft Skills |
| ✉️ **Cover Letter Generator** | One-click tailored cover letter powered by Groq LLaMA 3.3 70B |
| ✏️ **ATS Bullet Rewriter** | Rewrites weak resume bullets into 3 ATS-optimized versions with action verbs & metrics |
| 🎯 **Interview Prep** | 10 personalized predicted interview questions based on resume + JD skill gaps |
| 📄 **PDF Export** | Export the full analysis report as a downloadable PDF |

---

## 🏗️ Architecture

```
genai-resume-jd-matcher/
├── main.py                  ← FastAPI app — all routes & lifespan
├── app/
│   ├── parser.py            ← PDF/DOCX text extraction (PyMuPDF + python-docx)
│   ├── embedder.py          ← Groq LLM similarity scoring + TF-IDF fallback
│   ├── analyzer.py          ← Groq LLaMA 3.3 — analysis, cover letter, bullet rewrite, interview prep
│   └── models.py            ← Pydantic request/response schemas
├── templates/
│   └── index.html           ← Dark UI — radar chart, tabs, skill tags, drag-and-drop upload
├── static/                  ← Static assets
├── Dockerfile
├── render.yaml
└── requirements.txt
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/harshupadhyay14/GenAI-Resume-JD-Matching-Skill-Gap-Analyzer.git
cd GenAI-Resume-JD-Matching-Skill-Gap-Analyzer
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Add API Key
```bash
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
2. Go to [render.com](https://render.com) → **New → Web Service** → Connect repo
3. Add environment variable: `GROQ_API_KEY` = your key
4. Deploy — live in ~2 minutes

The `render.yaml` is pre-configured with the correct build and start commands.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/analyze` | Upload resume (PDF/DOCX) + JD → full match analysis |
| `POST` | `/api/cover-letter` | Generate tailored cover letter |
| `POST` | `/api/rewrite-bullet` | Rewrite a resume bullet in 3 ATS-optimized versions |
| `POST` | `/api/interview-questions` | Generate 10 predicted interview questions |
| `GET`  | `/api/health` | Health check |

### Example — Analyze Resume
```bash
curl -X POST "http://localhost:8000/api/analyze" \
  -F "resume=@resume.pdf" \
  -F "job_description=We are looking for a Python developer with FastAPI and ML experience..."
```

### Example Response
```json
{
  "match_score": 72.0,
  "matched_skills": ["Python", "FastAPI", "Machine Learning"],
  "missing_skills": ["Docker", "Kubernetes"],
  "overall_assessment": "Strong candidate with solid Python and ML foundation...",
  "category_scores": {
    "Technical Skills": 75,
    "Experience Relevance": 60,
    "Education": 70,
    "Projects": 80,
    "Soft Skills": 65
  }
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | FastAPI, Python 3.11, Uvicorn |
| **LLM (Similarity)** | Groq API — LLaMA 3.3 70B (semantic match scoring) |
| **LLM (Analysis)** | Groq API — LLaMA 3.3 70B (cover letter, bullet rewrite, interview prep) |
| **Fallback Similarity** | Scikit-learn TF-IDF + cosine similarity |
| **File Parsing** | PyMuPDF (PDF), python-docx (DOCX) |
| **Frontend** | Vanilla JS, Chart.js (radar chart), dark UI |
| **Deployment** | Render, Docker |

---

## 💡 How the Match Score Works

Unlike traditional keyword matchers, this app uses **Groq LLaMA 3.3 70B** to semantically understand both the resume and job description — the same way a human recruiter would. It considers:

- ✅ Skill alignment (exact + related skills)
- ✅ Experience relevance
- ✅ Project complexity
- ✅ Overall candidate fit

A **TF-IDF cosine similarity** fallback ensures the app always returns a score even without an API key.

---

## 📁 Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | Groq API key from [console.groq.com](https://console.groq.com) |

---

## 📄 License

MIT © Harsh Upadhyay
