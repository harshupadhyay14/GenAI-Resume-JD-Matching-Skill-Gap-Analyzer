"""
Embedder Module — Lightweight Version (HuggingFace API + TF-IDF fallback)
Fixes 0% match score by rescaling TF-IDF cosine similarity to 0–1 range.
"""

import re
import os
import numpy as np
import httpx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

HF_API_KEY = os.getenv("HF_API_KEY", "")
HF_API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"

SKILL_KEYWORDS = {
    "Programming Languages": [
        "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
        "kotlin", "swift", "ruby", "php", "scala", "r", "matlab", "dart", "bash"
    ],
    "Web Frameworks": [
        "react", "angular", "vue", "next.js", "django", "flask", "fastapi",
        "spring", "express", "node.js", "laravel", "rails", "svelte", "streamlit"
    ],
    "Data & ML": [
        "machine learning", "deep learning", "neural network", "tensorflow", "pytorch",
        "keras", "scikit-learn", "pandas", "numpy", "matplotlib", "seaborn",
        "xgboost", "lightgbm", "random forest", "regression", "classification",
        "clustering", "nlp", "computer vision", "feature engineering", "model evaluation"
    ],
    "Generative AI": [
        "llm", "gpt", "llama", "groq", "openai", "langchain", "rag",
        "retrieval augmented generation", "prompt engineering", "embeddings",
        "vector database", "fine-tuning", "hugging face", "transformers",
        "text generation", "semantic search", "faiss", "pinecone", "chroma"
    ],
    "Databases": [
        "sql", "mysql", "postgresql", "mongodb", "redis", "sqlite", "oracle",
        "cassandra", "dynamodb", "elasticsearch", "neo4j", "firebase"
    ],
    "Cloud & DevOps": [
        "aws", "azure", "gcp", "docker", "kubernetes", "ci/cd", "jenkins",
        "github actions", "terraform", "ansible", "ec2", "s3", "lambda",
        "render", "heroku", "vercel", "netlify"
    ],
    "Data Analytics & BI": [
        "power bi", "tableau", "excel", "dax", "power query", "eda",
        "exploratory data analysis", "pivot table", "xlookup", "data visualization",
        "looker", "metabase"
    ],
    "Tools & Practices": [
        "git", "github", "linux", "rest api", "graphql", "microservices",
        "agile", "scrum", "jira", "confluence", "postman", "jupyter", "vs code"
    ],
    "Soft Skills": [
        "communication", "leadership", "teamwork", "problem solving",
        "critical thinking", "project management", "collaboration", "mentoring"
    ],
}

ALL_SKILLS = []
for category, skills in SKILL_KEYWORDS.items():
    ALL_SKILLS.extend(skills)


def get_model():
    """No-op: kept for compatibility with main.py lifespan call."""
    print("Lightweight mode: using HuggingFace API + TF-IDF fallback.")


def _hf_embed(texts: list) -> np.ndarray:
    """Call HuggingFace Inference API for embeddings."""
    if not HF_API_KEY:
        return None
    try:
        headers = {"Authorization": f"Bearer {HF_API_KEY}"}
        with httpx.Client(timeout=20.0) as client:
            response = client.post(
                HF_API_URL,
                headers=headers,
                json={"inputs": texts, "options": {"wait_for_model": True}},
            )
        if response.status_code == 200:
            data = response.json()
            result = []
            for item in data:
                if isinstance(item[0], list):
                    arr = np.array(item)
                    result.append(arr.mean(axis=0))
                else:
                    result.append(np.array(item))
            return np.array(result, dtype="float32")
    except Exception as e:
        print(f"HF API error: {e}")
    return None


def _tfidf_similarity(text1: str, text2: str) -> float:
    """
    TF-IDF cosine similarity with rescaling.
    Raw TF-IDF scores on long docs are naturally very low (0.02–0.15).
    We rescale to a 0–1 range that maps meaningfully to 0–100% for display.
    Rescaling: score / 0.35 clamped to [0, 1]
    (0.35 is the practical max cosine sim between a resume and JD via TF-IDF)
    """
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        raw_score = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])

        # Rescale: raw TF-IDF cosine sim between resume/JD rarely exceeds 0.35
        # Map 0.0–0.35 → 0.0–1.0 for meaningful percentage display
        rescaled = raw_score / 0.35
        return float(max(0.0, min(1.0, rescaled)))
    except Exception as e:
        print(f"TF-IDF similarity error: {e}")
        return 0.5


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between resume and JD.
    Uses HuggingFace API if HF_API_KEY is set, otherwise rescaled TF-IDF.
    Returns float between 0 and 1.
    """
    # Try HuggingFace API first
    embeddings = _hf_embed([text1, text2])
    if embeddings is not None and len(embeddings) == 2:
        a, b = embeddings[0], embeddings[1]
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        score = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, min(1.0, score))

    # Fallback: rescaled TF-IDF
    return _tfidf_similarity(text1, text2)


def extract_skills_from_text(text: str) -> list:
    """Extract skills from text using keyword matching."""
    text_lower = text.lower()
    found_skills = []

    for skill in ALL_SKILLS:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.append(skill.title() if len(skill.split()) == 1 else skill.title())

    seen = set()
    unique_skills = []
    for s in found_skills:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique_skills.append(s)

    return unique_skills


def get_skills_by_category(skills: list) -> dict:
    """Group skills by their category."""
    categorized = {}
    skill_lower_map = {s.lower(): s for s in skills}

    for category, keyword_list in SKILL_KEYWORDS.items():
        matched = []
        for kw in keyword_list:
            if kw.lower() in skill_lower_map:
                matched.append(skill_lower_map[kw.lower()])
        if matched:
            categorized[category] = matched

    return categorized