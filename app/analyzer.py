"""
LLM Analyzer Module — Upgraded
Uses Groq LLaMA 3 for:
- Deep match analysis
- Cover letter generation
- Resume bullet rewriting
- Interview question prediction
"""

import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

ANALYSIS_PROMPT = """You are an expert technical recruiter and career coach with 15+ years of experience.

Analyze the following resume against the job description and provide a detailed, structured analysis.

=== RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{job_description}

=== SEMANTIC MATCH SCORE ===
{similarity_score}% (computed via sentence embeddings)

=== MATCHED SKILLS ===
{matched_skills}

=== MISSING SKILLS ===
{missing_skills}

Respond ONLY with this exact JSON (no markdown, no extra text):

{{
  "overall_assessment": "2-3 sentence overall assessment of candidate fit",
  "strengths": [
    "Specific strength 1 from resume",
    "Specific strength 2",
    "Specific strength 3",
    "Specific strength 4"
  ],
  "skill_gap_details": [
    {{
      "skill": "Skill name",
      "importance": "High/Medium/Low",
      "suggestion": "Specific actionable advice",
      "resource": "Free course or resource URL e.g. https://www.coursera.org/..."
    }}
  ],
  "improvement_suggestions": [
    "Specific improvement 1",
    "Specific improvement 2",
    "Specific improvement 3",
    "Specific improvement 4",
    "Specific improvement 5"
  ],
  "category_scores": {{
    "Technical Skills": 75,
    "Experience Relevance": 60,
    "Education": 70,
    "Projects": 80,
    "Soft Skills": 65
  }}
}}"""

COVER_LETTER_PROMPT = """You are an expert career coach. Write a compelling, personalized cover letter.

=== RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{job_description}

Write a professional cover letter (3-4 paragraphs) that:
1. Opens with a strong hook specific to this role
2. Highlights 2-3 most relevant experiences/projects from the resume
3. Addresses why this specific company/role excites the candidate
4. Closes with a confident call to action

Keep it under 350 words. Sound human, not generic. Infer company name from JD if possible, else use "your team".
Return ONLY the cover letter text, no JSON, no extra formatting."""

BULLET_REWRITER_PROMPT = """You are an expert resume coach. Rewrite the following resume bullet point to be stronger.

Original bullet: {bullet}
Job description context: {job_description}

Rules:
- Start with a strong action verb
- Add quantifiable metrics where possible
- Mirror keywords from the job description
- Keep under 20 words
- Make it ATS-friendly

Return ONLY a JSON array with exactly 3 improved versions:
["version 1", "version 2", "version 3"]"""

INTERVIEW_QUESTIONS_PROMPT = """You are a senior technical interviewer. Based on this resume and job description, predict the most likely interview questions.

=== RESUME ===
{resume_text}

=== JOB DESCRIPTION ===
{job_description}

=== SKILL GAPS ===
{missing_skills}

Generate 10 interview questions:
- 4 technical questions based on JD requirements
- 3 behavioral questions based on candidate's experience  
- 2 questions probing skill gaps
- 1 curveball question

Return ONLY this JSON (no markdown):
[
  {{
    "question": "Question text",
    "type": "Technical/Behavioral/Skill Gap/Curveball",
    "tip": "Brief answering tip in one sentence"
  }}
]"""


def analyze_match_with_llm(resume_text, job_description, similarity_score, matched_skills, missing_skills):
    prompt = ANALYSIS_PROMPT.format(
        resume_text=resume_text[:3000],
        job_description=job_description[:2000],
        similarity_score=round(similarity_score * 100, 1),
        matched_skills=", ".join(matched_skills) if matched_skills else "None",
        missing_skills=", ".join(missing_skills) if missing_skills else "None",
    )
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an expert technical recruiter. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        raw = re.sub(r"^```json\s*", "", response.choices[0].message.content.strip())
        raw = re.sub(r"\s*```$", "", raw)
        analysis = json.loads(raw)
        return {
            "overall_assessment": analysis.get("overall_assessment", "Analysis completed."),
            "strengths": analysis.get("strengths", []),
            "skill_gap_details": analysis.get("skill_gap_details", []),
            "improvement_suggestions": analysis.get("improvement_suggestions", []),
            "category_scores": analysis.get("category_scores", {
                "Technical Skills": 50, "Experience Relevance": 50,
                "Education": 50, "Projects": 50, "Soft Skills": 50,
            }),
        }
    except Exception as e:
        print(f"LLM analysis error: {e}")
        return _fallback_analysis(matched_skills, missing_skills, similarity_score)


def generate_cover_letter(resume_text: str, job_description: str) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an expert career coach who writes compelling, human-sounding cover letters."},
                {"role": "user", "content": COVER_LETTER_PROMPT.format(
                    resume_text=resume_text[:2500],
                    job_description=job_description[:1500]
                )}
            ],
            temperature=0.7,
            max_tokens=800,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error generating cover letter: {str(e)}"


def rewrite_bullet_points(bullet: str, job_description: str) -> list:
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are an expert resume coach. Return only valid JSON arrays."},
                {"role": "user", "content": BULLET_REWRITER_PROMPT.format(
                    bullet=bullet,
                    job_description=job_description[:800]
                )}
            ],
            temperature=0.7,
            max_tokens=400,
        )
        raw = re.sub(r"^```json\s*", "", response.choices[0].message.content.strip())
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        return [
            f"Engineered {bullet[:45]}... delivering measurable performance improvements",
            f"Architected and deployed {bullet[:40]}... achieving 30% efficiency gain",
            f"Led development of {bullet[:45]}... resulting in production-ready solution",
        ]


def generate_interview_questions(resume_text: str, job_description: str, missing_skills: list) -> list:
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a senior technical interviewer. Return only valid JSON."},
                {"role": "user", "content": INTERVIEW_QUESTIONS_PROMPT.format(
                    resume_text=resume_text[:2000],
                    job_description=job_description[:1500],
                    missing_skills=", ".join(missing_skills[:8])
                )}
            ],
            temperature=0.5,
            max_tokens=1500,
        )
        raw = re.sub(r"^```json\s*", "", response.choices[0].message.content.strip())
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        return [
            {"question": "Walk me through your most impactful ML project.", "type": "Technical", "tip": "Use STAR method and quantify results."},
            {"question": "How do you handle model drift in production?", "type": "Technical", "tip": "Mention monitoring and retraining pipelines."},
            {"question": "Describe a time you debugged a complex system issue.", "type": "Behavioral", "tip": "Focus on your systematic debugging approach."},
            {"question": "What's your experience with RAG systems?", "type": "Skill Gap", "tip": "Be honest and show eagerness to learn fast."},
        ]


def _fallback_analysis(matched_skills, missing_skills, similarity_score):
    score = round(similarity_score * 100)
    return {
        "overall_assessment": f"Semantic match score: {score}%. The resume shows overlap with the job description. Manual review recommended.",
        "strengths": [f"Matched skill: {s}" for s in matched_skills[:4]] or ["Resume uploaded successfully"],
        "skill_gap_details": [
            {"skill": s, "importance": "Medium", "suggestion": f"Consider adding {s} through online courses.", "resource": "https://www.coursera.org"}
            for s in missing_skills[:5]
        ],
        "improvement_suggestions": [
            "Tailor your resume to mirror keywords from the job description.",
            "Quantify achievements with metrics (e.g., 'reduced latency by 30%').",
            "Add relevant projects that demonstrate missing skills.",
            "Include certifications for high-priority missing skills.",
            "Update your LinkedIn profile to reflect all mentioned skills.",
        ],
        "category_scores": {
            "Technical Skills": min(100, score + 10),
            "Experience Relevance": score,
            "Education": 70,
            "Projects": min(100, score + 5),
            "Soft Skills": 65,
        },
    }