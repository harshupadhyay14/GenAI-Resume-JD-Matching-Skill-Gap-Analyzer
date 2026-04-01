"""
Pydantic Models — Upgraded
"""
from pydantic import BaseModel
from typing import Optional


class SkillGapDetail(BaseModel):
    skill: str
    importance: str
    suggestion: str
    resource: Optional[str] = ""


class MatchResponse(BaseModel):
    match_score: float
    resume_skills: list[str]
    jd_skills: list[str]
    matched_skills: list[str]
    missing_skills: list[str]
    overall_assessment: str
    strengths: list[str]
    skill_gap_details: list[SkillGapDetail]
    improvement_suggestions: list[str]
    category_scores: dict[str, int]


class CoverLetterRequest(BaseModel):
    resume_text: str
    job_description: str


class CoverLetterResponse(BaseModel):
    cover_letter: str


class BulletRewriteRequest(BaseModel):
    bullet: str
    job_description: str


class BulletRewriteResponse(BaseModel):
    rewrites: list[str]


class InterviewQuestionsResponse(BaseModel):
    questions: list[dict]