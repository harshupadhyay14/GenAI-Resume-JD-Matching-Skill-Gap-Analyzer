"""
GenAI Resume–JD Matching & Skill Gap Analyzer — Upgraded
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
from pathlib import Path
from dotenv import load_dotenv

from app.parser import extract_text_from_file
from app.embedder import compute_similarity, extract_skills_from_text
from app.analyzer import (
    analyze_match_with_llm,
    generate_cover_letter,
    rewrite_bullet_points,
    generate_interview_questions,
)
from app.models import (
    MatchResponse, CoverLetterRequest, CoverLetterResponse,
    BulletRewriteRequest, BulletRewriteResponse, InterviewQuestionsResponse
)

load_dotenv()

app = FastAPI(title="GenAI Resume–JD Matcher", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Store last analysis in memory for subsequent AI calls
_last_analysis_cache = {}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()


@app.post("/api/analyze", response_model=MatchResponse)
async def analyze_resume(
    resume: UploadFile = File(...),
    job_description: str = Form(...),
):
    file_ext = Path(resume.filename).suffix.lower()
    if file_ext not in [".pdf", ".docx"]:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    file_path = UPLOAD_DIR / resume.filename
    content = await resume.read()
    with open(file_path, "wb") as f:
        f.write(content)

    try:
        resume_text = extract_text_from_file(str(file_path))
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from file.")

        similarity_score = compute_similarity(resume_text, job_description)
        resume_skills = extract_skills_from_text(resume_text)
        jd_skills = extract_skills_from_text(job_description)
        matched_skills = [s for s in jd_skills if any(s.lower() in rs.lower() or rs.lower() in s.lower() for rs in resume_skills)]
        missing_skills = [s for s in jd_skills if s not in matched_skills]

        llm_analysis = analyze_match_with_llm(resume_text, job_description, similarity_score, matched_skills, missing_skills)

        # Cache for subsequent calls
        _last_analysis_cache["resume_text"] = resume_text
        _last_analysis_cache["job_description"] = job_description
        _last_analysis_cache["missing_skills"] = missing_skills

        return MatchResponse(
            match_score=round(similarity_score * 100, 2),
            resume_skills=resume_skills,
            jd_skills=jd_skills,
            matched_skills=matched_skills,
            missing_skills=missing_skills,
            overall_assessment=llm_analysis["overall_assessment"],
            improvement_suggestions=llm_analysis["improvement_suggestions"],
            skill_gap_details=llm_analysis["skill_gap_details"],
            strengths=llm_analysis["strengths"],
            category_scores=llm_analysis["category_scores"],
        )
    finally:
        if file_path.exists():
            os.remove(file_path)


@app.post("/api/cover-letter", response_model=CoverLetterResponse)
async def get_cover_letter(req: CoverLetterRequest):
    """Generate a tailored cover letter."""
    letter = generate_cover_letter(req.resume_text, req.job_description)
    return CoverLetterResponse(cover_letter=letter)


@app.post("/api/rewrite-bullet", response_model=BulletRewriteResponse)
async def rewrite_bullet(req: BulletRewriteRequest):
    """Rewrite a resume bullet point in 3 improved versions."""
    rewrites = rewrite_bullet_points(req.bullet, req.job_description)
    return BulletRewriteResponse(rewrites=rewrites)


@app.post("/api/interview-questions", response_model=InterviewQuestionsResponse)
async def get_interview_questions(req: CoverLetterRequest):
    """Generate predicted interview questions."""
    missing = _last_analysis_cache.get("missing_skills", [])
    questions = generate_interview_questions(req.resume_text, req.job_description, missing)
    return InterviewQuestionsResponse(questions=questions)


@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)