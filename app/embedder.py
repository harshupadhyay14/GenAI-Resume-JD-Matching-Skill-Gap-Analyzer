"""
Embedder Module — Groq-only / Lightweight Version
Uses TF-IDF + cosine similarity instead of local sentence-transformers.
Groq does not offer an embeddings API, so TF-IDF with bigrams is used —
fast, ~5MB RAM, and works well for keyword-rich resume/JD text.
"""

import re
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Comprehensive skill keyword list across 10+ categories
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
    print("Lightweight mode: using TF-IDF similarity (no local model loaded).")


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using TF-IDF.
    Uses unigrams + bigrams for better phrase matching.
    Returns a float between 0 and 1.
    """
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=8000,
            ngram_range=(1, 2),
        )
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
        return float(max(0.0, min(1.0, score)))
    except Exception as e:
        print(f"Similarity computation error: {e}")
        return 0.5


def extract_skills_from_text(text: str) -> list[str]:
    """
    Extract skills from text using keyword matching.
    Returns a deduplicated list of found skills.
    """
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


def get_skills_by_category(skills: list[str]) -> dict[str, list[str]]:
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