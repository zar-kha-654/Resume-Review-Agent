import time

import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, LLM


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "groq/openai/gpt-oss-120b"


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_text(uploaded_file):
    """Extract readable text from an uploaded PDF."""

    try:
        reader = PdfReader(uploaded_file)

        if not reader.pages:
            raise ValueError("The PDF does not contain any pages.")

        extracted_text = []

        for page in reader.pages:
            text = page.extract_text()

            if text:
                extracted_text.append(text)

        final_text = "\n".join(extracted_text).strip()

        if not final_text:
            raise ValueError(
                "No readable text was found in this PDF. "
                "The PDF may be scanned or image-based."
            )

        return final_text

    except Exception as e:
        raise ValueError(
            f"Could not extract text from PDF: {str(e)}"
        )


# ============================================================
# GET GROQ API KEY FROM STREAMLIT SECRETS
# ============================================================

def get_groq_api_key():
    """Get the Groq API key from Streamlit Secrets."""

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            return None

        return api_key

    except Exception:
        return None


# ============================================================
# CREATE GROQ LLM
# ============================================================

def create_llm(api_key):
    """Create the CrewAI LLM using Groq."""

    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.1,
        max_tokens=5000
    )


# ============================================================
# CREATE RESUME REVIEW AGENT
# ============================================================

def create_agent(llm):
    """Create the single CrewAI resume reviewer."""

    return Agent(
        role="Professional Resume Reviewer",

        goal=(
            "Analyze a candidate's resume against a target job "
            "description and provide an accurate, evidence-based "
            "and actionable review without inventing qualifications."
        ),

        backstory=(
            "You are an experienced professional resume reviewer "
            "and career-document analyst. You carefully compare "
            "resumes with job descriptions. You only consider "
            "qualifications that are explicitly stated or clearly "
            "demonstrated in the resume. You never assume that a "
            "candidate has a skill, certification, degree, job "
            "experience, or achievement that is not supported by "
            "the resume."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False
    )


# ============================================================
# CREATE REVIEW TASK
# ============================================================

def create_task(agent, resume_text, job_description):
    """Create the resume review task."""

    task_description = f"""
You are reviewing a candidate's resume for a target job.

========================
CANDIDATE RESUME
========================

{resume_text}


========================
TARGET JOB DESCRIPTION
========================

{job_description}


========================
IMPORTANT RULES
========================

1. Do NOT fabricate qualifications.

2. Only claim that the candidate has a skill, technology,
certification, degree, achievement, or experience when it is
explicitly supported by the resume.

3. If something is required by the job description but is not
mentioned in the resume, say:

"Not mentioned in the resume."

4. Do not interpret silence as possession.

5. Do not claim that the candidate lacks a skill merely because
the skill is not mentioned. Say that it was not mentioned.

6. Do not invent years of experience.

7. Do not invent projects, employers, degrees, certifications,
achievements, or technologies.

8. Do not rewrite the candidate's experience as if they had
experience they did not state.

9. Base important conclusions on evidence from the supplied
resume and job description.

10. Recommendations should be practical and actionable.

11. Suggested keywords should only be recommended if the
candidate genuinely has the corresponding skill or experience.

========================
OUTPUT FORMAT
========================

# Resume Match Review

## 1. Overall Match

Give a short summary explaining how closely the resume aligns
with the target job.

Do not make unsupported claims about the candidate's ability.

## 2. Requirements Found in the Resume

List important job requirements that are explicitly supported
by the resume.

For each item include:

- Requirement
- Resume evidence
- Match status

Use:

- Strong match
- Partial match

## 3. Requirements Not Mentioned

List important job requirements that are not found in the resume.

For each item include:

- Job requirement
- Resume status: Not mentioned in the resume
- Why it matters

Do not claim that the candidate lacks the skill.

## 4. Experience Alignment

Explain how the candidate's stated work experience, projects,
internships, education, or other experience relates to the job.

Only use information actually present in the resume.

## 5. Resume Problems

Identify issues such as:

- unclear bullet points
- weak descriptions
- missing measurable results
- irrelevant information
- unclear technical skills
- poor organization
- missing relevant keywords

Do not invent achievements.

## 6. Actionable Improvements

Give specific improvements the candidate can make.

When suggesting stronger wording, clearly indicate that the
candidate should only use it if it is truthful.

## 7. Suggested Keywords

List relevant keywords from the job description that the
candidate could consider adding ONLY if they genuinely have
those skills or experience.

## 8. Interview Preparation Areas

List topics the candidate should prepare for based on the
overlap between the resume and job description.

## 9. Final Action Plan

Give 5 to 8 practical next steps in priority order.

Keep everything evidence-based.
"""

    return Task(
        description=task_description,

        expected_output=(
            "A structured resume review containing overall "
            "alignment, supported requirements, requirements "
            "not mentioned, experience alignment, resume "
            "problems, actionable improvements, suggested "
            "keywords, interview preparation areas, and "
            "a final action plan."
        ),

        agent=agent
    )


# ============================================================
# RUN RESUME REVIEW
# ============================================================

def run_resume_review(resume_text, job_description, api_key):
    """Run the CrewAI resume review with error handling."""

    llm = create_llm(api_key)

    agent = create_agent(llm)

    task = create_task(
        agent,
        resume_text,
        job_description
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )

    max_attempts = 3

    for attempt in range(max_attempts):

        try:

            result = crew.kickoff()

            if result is None:
                raise RuntimeError(
                    "The AI returned an empty response."
                )

            return str(result)

        except Exception as e:

            error_text = str(e).lower()

            # --------------------------------------------
            # RATE LIMIT
            # --------------------------------------------

            if (
                "429" in error_text
                or "rate limit" in error_text
                or "ratelimit" in error_text
            ):

                if attempt < max_attempts - 1:

                    wait_time = 2 ** attempt

                    time.sleep(wait_time)

                    continue

                raise RuntimeError(
                    "Groq rate limit reached. "
                    "Please wait a little and try again."
                )

            # --------------------------------------------
            # API KEY / AUTHENTICATION
            # --------------------------------------------

            if (
                "401" in error_text
                or "authentication" in error_text
                or "unauthorized" in error_text
                or "api key" in error_text
            ):

                raise RuntimeError(
                    "Groq API authentication failed. "
                    "Please check your GROQ_API_KEY in "
                    "Streamlit Secrets."
                )

            # --------------------------------------------
            # PERMISSION
            # --------------------------------------------

            if (
                "403" in error_text
                or "permission" in error_text
                or "model_permission" in error_text
            ):

                raise RuntimeError(
                    "The selected Groq model is not available "
                    "for this API project."
                )

            # --------------------------------------------
            # MODEL ERROR
            # --------------------------------------------

            if (
                "model not found" in error_text
                or "does not exist" in error_text
                or "decommissioned" in error_text
            ):

                raise RuntimeError(
                    f"The configured model '{MODEL_NAME}' "
                    "is unavailable."
                )

            # --------------------------------------------
            # TEMPORARY CONNECTION ERROR
            # --------------------------------------------

            if (
                "connection" in error_text
                or "timeout" in error_text
                or "temporarily unavailable" in error_text
                or "503" in error_text
                or "502" in error_text
            ):

                if attempt < max_attempts - 1:

                    time.sleep(2 ** attempt)

                    continue

                raise RuntimeError(
                    "The Groq service could not be reached. "
                    "Please try again."
                )

            # --------------------------------------------
            # OTHER ERROR
            # --------------------------------------------

            raise RuntimeError(
                f"The AI review failed: {str(e)}"
            )

    raise RuntimeError(
        "The resume review could not be completed."
    )


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("📄 AI Resume Reviewer")

st.write(
    "Compare a resume with a target job description "
    "using a CrewAI-powered AI reviewer."
)

st.info(
    "The reviewer only uses information provided in the "
    "resume and does not assume unmentioned qualifications."
)


# ============================================================
# RESUME INPUT
# ============================================================

st.header("1. Candidate Resume")

resume_input_method = st.radio(
    "Choose resume input method:",
    ["Upload PDF", "Paste Resume Text"],
    horizontal=T
```
