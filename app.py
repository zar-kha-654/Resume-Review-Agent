```python
import time
import streamlit as st
from pypdf import PdfReader
from crewai import Agent, Task, Crew, LLM


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Reviewer",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

MODEL_NAME = "groq/openai/gpt-oss-120b"


# ============================================================
# HELPER: EXTRACT TEXT FROM PDF
# ============================================================

def extract_pdf_text(uploaded_file):
    """
    Extract text from an uploaded PDF.

    Returns:
        str: Extracted text
    """

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
        raise ValueError(f"Could not extract text from PDF: {str(e)}")


# ============================================================
# HELPER: GET GROQ API KEY
# ============================================================

def get_groq_api_key():
    """
    Read the Groq API key from Streamlit secrets.
    """

    try:
        api_key = st.secrets["GROQ_API_KEY"]

        if not api_key:
            return None

        return api_key

    except Exception:
        return None


# ============================================================
# CREATE CREWAI LLM
# ============================================================

def create_llm(api_key):
    """
    Create the CrewAI LLM using Groq.
    """

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
    """
    Create the single resume-review agent.
    """

    return Agent(
        role="Professional Resume Reviewer",

        goal=(
            "Analyze a candidate's resume against a target job description "
            "and provide an accurate, evidence-based and actionable review "
            "without inventing qualifications."
        ),

        backstory=(
            "You are an experienced professional resume reviewer and "
            "career-document analyst. You carefully compare resumes with "
            "job descriptions. You only consider qualifications that are "
            "explicitly stated or clearly demonstrated in the resume. "
            "You never assume that a candidate has a skill, certification, "
            "degree, job experience or achievement that is not supported "
            "by the resume."
        ),

        llm=llm,

        verbose=False,

        allow_delegation=False
    )


# ============================================================
# BUILD REVIEW TASK
# ============================================================

def create_task(agent, resume_text, job_description):
    """
    Create the task given to the CrewAI agent.
    """

    task_description = f"""
You are reviewing a candidate's resume for a specific target job.

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
   certification, degree, achievement or experience when it is
   explicitly supported by the resume.

3. If something is required by the job description but is not
   mentioned in the resume, say:

   "Not mentioned in the resume."

4. Do not interpret silence as possession.

5. Do not punish the candidate simply because something is not
   mentioned. Clearly distinguish between:

   - explicitly present
   - partially demonstrated
   - not mentioned
   - conflicting information

6. Do not invent years of experience.

7. Do not invent projects, employers, degrees or certifications.

8. Do not rewrite the candidate's experience as if they had
   experience they did not state.

9. Base every important conclusion on evidence from the supplied
   resume and job description.

10. The purpose is to help the candidate improve the resume,
    not to make unsupported judgments about the candidate.

========================
OUTPUT FORMAT
========================

Produce the report using exactly these sections:

# Resume Match Review

## 1. Overall Match

Give a short summary of how closely the resume aligns with the
job description.

Do NOT invent a percentage unless you can clearly justify it from
the evidence. If you provide a score, explain that it is an
approximate document-to-job-description match rather than a measure
of the candidate's actual ability.

## 2. Requirements Found in the Resume

List important job requirements that are explicitly supported
by the resume.

For each item include:

- Requirement
- Resume evidence
- Match status

Use statuses such as:

- Strong match
- Partial match

## 3. Requirements Not Mentioned

List important job requirements that are not found in the resume.

For each item explain:

- Job requirement
- Resume status: Not mentioned in the resume
- Why it matters

Do NOT claim that the candidate lacks the skill.
Only say that it was not found in the supplied resume.

## 4. Experience Alignment

Explain how the candidate's stated work, project, internship,
education or other experience relates to the target role.

Only use experience actually present in the resume.

## 5. Resume Problems

Identify issues such as:

- unclear bullet points
- missing measurable results
- weak descriptions
- irrelevant information
- unclear technical skills
- formatting/content organization issues
- missing keywords that are actually relevant to the job

Do not invent missing achievements.

## 6. Actionable Improvements

Give specific improvements the candidate can make.

For example:

Instead of:

"Worked on Python projects."

Suggest:

"If accurate, rewrite the bullet to describe the specific Python
project, technology used, and measurable result."

Never invent the result for the candidate.

## 7. Suggested Keywords

List relevant keywords from the job description that the candidate
could consider adding ONLY if they genuinely have that skill or
experience.

## 8. Interview Preparation Areas

List topics the candidate should be prepared to discuss based on
the overlap between the resume and job description.

## 9. Final Action Plan

Give the candidate 5-8 concrete next steps in priority order.

Keep the recommendations practical and evidence-based.
"""

    return Task(
        description=task_description,

        expected_output=(
            "A structured resume review containing overall alignment, "
            "supported requirements, requirements not mentioned, "
            "experience alignment, resume problems, actionable "
            "improvements, suggested keywords, interview preparation "
            "areas and a final action plan."
        ),

        agent=agent
    )


# ============================================================
# RUN CREW WITH ERROR HANDLING
# ============================================================

def run_resume_review(resume_text, job_description, api_key):
    """
    Run the CrewAI resume reviewer.

    Includes handling for common API/runtime failures.
    """

    try:

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

        # Small retry mechanism for temporary failures.
        max_attempts = 3

        for attempt in range(max_attempts):

            try:

                result = crew.kickoff()

                if result is None:
                    raise RuntimeError(
                        "The agent returned an empty response."
                    )

                return str(result)

            except Exception as e:

                error_text = str(e).lower()

                # Rate-limit errors
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

                # Authentication errors
                if (
                    "401" in error_text
                    or "authentication" in error_text
                    or "api key" in error_text
                    or "unauthorized" in error_text
                ):

                    raise RuntimeError(
                        "Groq API authentication failed. "
                        "Please check your GROQ_API_KEY."
                    )

                # Permission/model errors
                if (
                    "403" in error_text
                    or "permission" in error_text
                    or "model_permission" in error_text
                ):

                    raise RuntimeError(
                        "The selected Groq model is not available "
                        "for this API project."
                    )

                # Model errors
                if (
                    "model_decommissioned" in error_text
                    or "model not found" in error_text
                    or "does not exist" in error_text
                ):

                    raise RuntimeError(
                        f"The configured model "
                        f"'{MODEL_NAME}' is unavailable."
                    )

                # Temporary connection/API errors
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

                # Unknown error
                raise RuntimeError(
                    f"The AI review failed: {str(e)}"
                )

        raise RuntimeError("The review could not be completed.")

    except Exception as e:

        raise RuntimeError(str(e))


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

st.title("📄 AI Resume Reviewer")

st.write(
    "Compare a resume with a target job description using "
    "a CrewAI-powered AI reviewer."
)

st.info(
    "The reviewer only uses information provided in the resume "
    "and does not assume unmentioned qualifications."
)


# ============================================================
# INPUT SECTION
# ============================================================

st.header("1. Candidate Resume")

resume_input_method = st.radio(
    "Choose resume input method:",
    ["Upload PDF", "Paste Resume Text"],
    horizontal=True
)


resume_text = ""


if resume_input_method == "Upload PDF":

    uploaded_file = st.file_uploader(
        "Upload your resume PDF",
        type=["pdf"],
        help="Upload a text-based PDF resume."
    )

    if uploaded_file is not None:

        try:

            resume_text = extract_pdf_text(uploaded_file)

            st.success("Resume PDF successfully extracted.")

            with st.expander("Preview extracted resume text"):

                preview = resume_text[:5000]

                st.text(preview)

                if len(resume_text) > 5000:
                    st.caption(
                        "Preview truncated. The full extracted text "
                        "will be sent to the reviewer."
                    )

        except ValueError as e:

            st.error(str(e))

else:

    resume_text = st.text_area(
        "Paste your resume text here",
        height=350,
        placeholder=(
            "Paste the complete resume text here..."
        )
    )


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header("2. Target Job Description")

job_description = st.text_area(
    "Paste the job description here",
    height=350,
    placeholder=(
        "Paste the complete target job description here..."
    )
)


# ============================================================
# API KEY CHECK
# ============================================================

api_key = get_groq_api_key()


if not api_key:

    st.warning(
        "Groq API key is not configured. "
        "Add GROQ_API_KEY to Streamlit secrets before running a review."
    )


# ============================================================
# REVIEW BUTTON
# ============================================================

st.header("3. Analyze")

analyze_button = st.button(
    "🔍 Review Resume",
    type="primary",
    use_container_width=True
)


if analyze_button:

    # ----------------------------------------
    # Validate resume
    # ----------------------------------------

    if not resume_text.strip():

        st.error(
            "Please upload a resume PDF or paste resume text."
        )

        st.stop()


    # ----------------------------------------
    # Validate job description
    # ----------------------------------------

    if not job_description.strip():

        st.error(
            "Please paste the target job description."
        )

        st.stop()


    # ----------------------------------------
    # Validate API key
    # ----------------------------------------

    if not api_key:

        st.error(
            "Groq API key is missing. "
            "Configure GROQ_API_KEY in Streamlit secrets."
        )

        st.stop()


    # ----------------------------------------
    # Basic input-size protection
    # ----------------------------------------

    if len(resume_text) > 100000:

        st.error(
            "The resume text is unusually large. "
            "Please upload a shorter resume."
        )

        st.stop()


    if len(job_description) > 100000:

        st.error(
            "The job description is unusually large. "
            "Please provide a shorter job description."
        )

        st.stop()


    # ----------------------------------------
    # Run AI review
    # ----------------------------------------

    with st.spinner(
        "🤖 CrewAI is reviewing the resume..."
    ):

        try:

            report = run_resume_review(
                resume_text=resume_text,
                job_description=job_description,
                api_key=api_key
            )

            st.success("Resume review completed!")

            st.markdown("---")

            st.header("📊 Resume Review")

            st.markdown(report)

            # --------------------------------
            # Download report
            # --------------------------------

            st.download_button(
                label="⬇️ Download Review",
                data=report,
                file_name="resume_review.txt",
                mime="text/plain"
            )

        except RuntimeError as e:

            st.error(str(e))

        except Exception:

            st.error(
                "Something unexpected happened while "
                "processing the resume. Please try again."
            )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Powered by Streamlit + CrewAI + Groq GPT-OSS 120B"
)
```
