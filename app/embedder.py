"""
Embedder Module
Uses Sentence Transformers + FAISS for semantic similarity computation.
Also extracts skills using keyword matching + NLP heuristics.
"""

import re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Load model once at startup (cached globally)
_model = None

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

# Flatten for quick lookup
ALL_SKILLS = []
for category, skills in SKILL_KEYWORDS.items():
    ALL_SKILLS.extend(skills)


def get_model() -> SentenceTransformer:
    """Lazy-load the sentence transformer model."""
    global _model
    if _model is None:
        print("Loading SentenceTransformer model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
        print("Model loaded successfully.")
    return _model


def get_embedding(text: str) -> np.ndarray:
    """Get sentence embedding as a numpy array."""
    model = get_model()
    embedding = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
    return embedding.astype("float32")


def compute_similarity(text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using FAISS.
    Returns a float between 0 and 1.
    """
    emb1 = get_embedding(text1)
    emb2 = get_embedding(text2)

    # Build FAISS index with text1's embedding
    dim = emb1.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product on normalized = cosine similarity
    index.add(emb1)

    # Search for nearest neighbor (text2)
    distances, _ = index.search(emb2, k=1)
    similarity = float(distances[0][0])

    # Clamp to [0, 1]
    return max(0.0, min(1.0, similarity))


def extract_skills_from_text(text: str) -> list[str]:
    """
    Extract skills from text using keyword matching.
    Returns a deduplicated list of found skills (original casing from keyword list).
    """
    text_lower = text.lower()
    found_skills = []

    for skill in ALL_SKILLS:
        # Use word boundary matching for accuracy
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, text_lower):
            # Return proper-cased version
            found_skills.append(skill.title() if len(skill.split()) == 1 else skill.title())

    # Deduplicate while preserving order
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