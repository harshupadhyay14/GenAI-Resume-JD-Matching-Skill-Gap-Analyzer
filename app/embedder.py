"""
Embedder Module — Groq-powered Similarity
Uses Groq LLaMA to compute semantic similarity score between resume and JD.
Falls back to rescaled TF-IDF if Groq is unavailable.
No local model, no HuggingFace — zero heavy dependencies.
"""

import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
    print("Lightweight mode: using Groq LLM similarity + TF-IDF fallback.")


def _groq_similarity(text1: str, text2: str) -> float:
    """
    Use Groq LLaMA to estimate semantic match score between resume and JD.
    Returns a float between 0 and 1.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    if not groq_key:
        print("No GROQ_API_KEY found, using TF-IDF fallback.")
        return None
    try:
        from groq import Groq
        client = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert resume screener. "
                        "Your task is to rate how well a resume matches a job description. "
                        "Consider skills, experience, projects, and overall fit. "
                        "Respond with ONLY a single integer between 0 and 100. "
                        "No explanation, no text, no punctuation — just the number."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"Rate the match between this resume and job description from 0 to 100.\n\n"
                        f"RESUME:\n{text1[:2500]}\n\n"
                        f"JOB DESCRIPTION:\n{text2[:1500]}\n\n"
                        f"Reply with only a number 0-100:"
                    )
                }
            ],
            temperature=0.1,
            max_tokens=5,
        )
        raw = response.choices[0].message.content.strip()
        score = float(''.join(filter(lambda c: c.isdigit() or c == '.', raw)))
        score = max(0.0, min(100.0, score))
        print(f"Groq similarity score: {score}")
        return score / 100.0
    except Exception as e:
        print(f"Groq similarity error: {e}")
        return None


def _tfidf_similarity(text1: str, text2: str) -> float:
    """
    TF-IDF cosine similarity with rescaling as fallback.
    Raw scores are very low (0.02–0.15), rescaled to 0–1 range.
    """
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        raw_score = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])
        rescaled = raw_score / 0.35
        print(f"TF-IDF fallback score: {round(rescaled * 100, 1)}")
        return float(max(0.0, min(1.0, rescaled)))
    except Exception as e:
        print(f"TF-IDF similarity error: {e}")
        return 0.5


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute similarity between resume and JD.
    1. Try Groq LLaMA (semantic, accurate)
    2. Fall back to rescaled TF-IDF
    Returns float between 0 and 1.
    """
    score = _groq_similarity(text1, text2)
    if score is not None:
        return score
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