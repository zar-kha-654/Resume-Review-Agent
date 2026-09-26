import time

import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, LLM


# -----------------------------
# Configuration
# -----------------------------

MODEL_NAME = "groq/openai/gpt-oss-120b"

st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)


# -----------------------------
# PDF extraction
# -----------------------------

def extract_pdf_text(uploaded_file):
    try:
        reader = PdfReader(uploaded_file)

        if len(reader.pages) == 0:
            return None, "The PDF contains no pages."

        text_parts = []

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text_parts.append(page_text)

        text = "\n".join(text_parts).strip()

        if not text:
            return None, (
                "No readable text was found in this PDF. "
                "It may be a scanned/image-based PDF."
            )

        return text, None

    except Exception as e:
        return None, f"Could not read the PDF: {e}"


# -----------------------------
# Get API key
# -----------------------------

def get_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


# -----------------------------
# Create LLM
# -----------------------------

def create_llm(api_key):
    return LLM(
        model=MODEL_NAME,
        api_key=api_key,
        temperature=0.1,
        max_tokens=5000
    )


# -----------------------------
# Create agent
# -----------------------------

def create_agent(llm):
    return Agent(
        role="Professional Resume Reviewer",

        goal=(
            "Compare a candidate's resume with a target job "
            "description and provide accurate, evidence-based "
            "and actionable feedback."
        ),

        backstory=(
            "You are an experienced resume reviewer. "
            "You carefully compare resumes with job descriptions. "
            "You never invent skills, qualifications, experience, "
            "certifications, degrees, projects, or achievements. "
            "If something is not mentioned in the resume, you say "
            "that it is not mentioned rather than assuming the "
            "candidate does not have it."
        ),

        llm=llm,
        verbose=False,
        allow_delegation=False,
        cache=False
    )

# -----------------------------
# Create task
# -----------------------------

def create_review_task(agent, resume, job_description):

    task_description = f"""
Review the following candidate resume against the target job.

================ RESUME ================

{resume}

================ JOB DESCRIPTION ================

{job_description}

================ RULES ================

1. Never fabricate qualifications.

2. Only say the candidate has a skill when the resume explicitly
mentions or clearly demonstrates that skill.

3. If a job requirement is not found in the resume, write:
"Not mentioned in the resume."

4. Do not say that the candidate lacks a skill simply because
the skill is not mentioned.

5. Do not invent years of experience.

6. Do not invent employers, projects, degrees, certifications,
technologies, achievements, or responsibilities.

7. Recommendations must be truthful and actionable.

8. If suggesting a keyword, tell the candidate to add it only
if they genuinely have that skill or experience.

================ REPORT FORMAT ================

# Resume Match Review

## 1. Overall Match

Give a short evidence-based summary of how the resume aligns
with the job description.

## 2. Matching Requirements

List important job requirements that are supported by the resume.

For each one provide:

- Requirement
- Resume evidence
- Match status

Use "Strong match" or "Partial match".

## 3. Requirements Not Mentioned

List important job requirements that were not found in the resume.

For each one provide:

- Requirement
- Status: Not mentioned in the resume
- Why it matters

Do not claim that the candidate lacks the skill.

## 4. Experience Alignment

Explain how the candidate's stated experience, projects,
education, or internships relate to the target role.

Only use information from the resume.

## 5. Resume Problems

Identify issues such as:

- unclear bullet points
- weak descriptions
- missing measurable results
- irrelevant information
- unclear skills
- poor organization
- missing relevant keywords

Do not invent achievements.

## 6. Actionable Improvements

Give specific improvements the candidate can make.

Do not create fake achievements or experience.

## 7. Suggested Keywords

List relevant keywords from the job description that the
candidate should consider adding only if they genuinely
possess those skills.

## 8. Interview Preparation

List topics the candidate should prepare based on the
resume and job description.

## 9. Final Action Plan

Give 5 to 8 practical next steps.

Keep the entire review factual and evidence-based.
"""

    return Task(
        description=task_description,

        expected_output=(
            "A structured resume review with overall match, "
            "matching requirements, requirements not mentioned, "
            "experience alignment, resume problems, improvements, "
            "keywords, interview preparation, and action plan."
        ),

        agent=agent
    )


# -----------------------------
# Run CrewAI
# -----------------------------

def review_resume(resume, job_description, api_key):

    llm = create_llm(api_key)

    agent = create_agent(llm)

    task = create_review_task(
        agent,
        resume,
        job_description
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )

    for attempt in range(3):

        try:
            result = crew.kickoff()

            if result:
                return str(result)

            raise RuntimeError(
                "The AI returned an empty response."
            )

        except Exception as e:

            error = str(e).lower()

            if (
                "429" in error
                or "rate limit" in error
                or "ratelimit" in error
            ):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                raise RuntimeError(
                    "Groq rate limit reached. "
                    "Please wait and try again."
                )

            if (
                "401" in error
                or "unauthorized" in error
                or "authentication" in error
                or "api key" in error
            ):
                raise RuntimeError(
                    "Groq API authentication failed. "
                    "Check GROQ_API_KEY in Streamlit Secrets."
                )

            if "403" in error or "permission" in error:
                raise RuntimeError(
                    "The Groq model is not available for this API key."
                )

            if (
                "timeout" in error
                or "connection" in error
                or "503" in error
                or "502" in error
            ):
                if attempt < 2:
                    time.sleep(2 ** attempt)
                    continue

                raise RuntimeError(
                    "The Groq service is temporarily unavailable. "
                    "Please try again."
                )

            raise RuntimeError(
                f"The resume review failed: {e}"
            )

    raise RuntimeError(
        "The resume review could not be completed."
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.title("📄 AI Resume Reviewer")

st.write(
    "Compare a resume with a target job description "
    "using CrewAI and Groq."
)

st.info(
    "The AI will not assume qualifications that are not "
    "mentioned in the resume."
)


# -----------------------------
# Resume input
# -----------------------------

st.header("1. Candidate Resume")

input_method = st.radio(
    "How do you want to provide the resume?",
    ["Upload PDF", "Paste Text"],
    horizontal=True
)

resume_text = ""


if input_method == "Upload PDF":

    uploaded_file = st.file_uploader(
        "Upload resume PDF",
        type=["pdf"]
    )

    if uploaded_file is not None:

        resume_text, pdf_error = extract_pdf_text(
            uploaded_file
        )

        if pdf_error:
            st.error(pdf_error)

        else:
            st.success("PDF successfully read.")

            with st.expander("Preview resume text"):
                st.text(resume_text[:5000])


else:

    resume_text = st.text_area(
        "Paste resume text",
        height=350,
        placeholder="Paste the candidate's resume here..."
    )


# -----------------------------
# Job description
# -----------------------------

st.header("2. Target Job Description")

job_description = st.text_area(
    "Paste the job description",
    height=350,
    placeholder="Paste the complete job description here..."
)


# -----------------------------
# API key
# -----------------------------

api_key = get_api_key()

if not api_key:
    st.warning(
        "GROQ_API_KEY is not configured in Streamlit Secrets."
    )


# -----------------------------
# Review button
# -----------------------------

st.header("3. Review")

if st.button(
    "🔍 Review Resume",
    type="primary",
    use_container_width=True
):

    if not resume_text or not resume_text.strip():
        st.error(
            "Please upload a PDF or paste the resume text."
        )
        st.stop()

    if not job_description or not job_description.strip():
        st.error(
            "Please paste the target job description."
        )
        st.stop()

    if not api_key:
        st.error(
            "GROQ_API_KEY is missing from Streamlit Secrets."
        )
        st.stop()

    if len(resume_text) > 100000:
        st.error(
            "The resume is too large. Please use a shorter resume."
        )
        st.stop()

    if len(job_description) > 100000:
        st.error(
            "The job description is too large."
        )
        st.stop()

    with st.spinner(
        "🤖 CrewAI is reviewing the resume..."
    ):

        try:

            report = review_resume(
                resume_text,
                job_description,
                api_key
            )

            st.success(
                "Resume review completed!"
            )

            st.markdown("---")

            st.markdown(report)

            st.download_button(
                "⬇️ Download Review",
                data=report,
                file_name="resume_review.txt",
                mime="text/plain"
            )

        except Exception as e:

            st.error(str(e))


# -----------------------------
# Footer
# -----------------------------

st.markdown("---")

st.caption(
    "Powered by Streamlit + CrewAI + Groq"
)
