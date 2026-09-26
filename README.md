# 🤖 AI Resume Reviewer

An AI-powered resume review application built with **Streamlit, CrewAI, and Groq**.

The application compares a candidate's resume against a target job description and provides structured, actionable feedback to help improve the resume for the specific role.

## ✨ Features

* 📄 Upload a resume as a PDF
* 📝 Paste resume text directly
* 💼 Enter a target job description
* 🤖 AI-powered resume analysis using CrewAI
* ⚡ Powered by Groq's `openai/gpt-oss-120b` model
* 🎯 Identifies matching job requirements
* 🔍 Identifies requirements not mentioned in the resume
* 📊 Analyzes experience alignment
* 🛠️ Provides actionable resume improvements
* 🔑 Uses Streamlit Secrets for secure API-key management
* 📥 Download the generated review as a text file

## 🏗️ Tech Stack

* **Python 3.11**
* **Streamlit** — Web application interface
* **CrewAI** — AI agent framework
* **Groq** — Fast LLM inference
* **GPT-OSS 120B** — Language model
* **PyPDF** — PDF text extraction
* **LiteLLM** — LLM integration layer

## 📁 Project Structure

```text
resume-review-agent/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd resume-review-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add your Groq API key

For local testing, create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

Do **not** commit your API key to GitHub.

## 🚀 Run Locally

Start the application with:

```bash
streamlit run app.py
```

The application will open in your browser.

## ☁️ Deploy on Streamlit Community Cloud

1. Push the project to GitHub.
2. Open Streamlit Community Cloud.
3. Select your GitHub repository.
4. Select `app.py` as the main file.
5. Use **Python 3.11**.
6. Open **Advanced Settings → Secrets**.
7. Add:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

8. Deploy the application.

## 🔐 Security

The Groq API key is loaded using:

```python
st.secrets["GROQ_API_KEY"]
```

The API key should never be hard-coded in `app.py` or committed to GitHub.

## 🧠 How It Works

```text
Resume PDF / Text
        ↓
Resume Text Extraction
        ↓
        +
Job Description
        ↓
CrewAI Resume Reviewer Agent
        ↓
Groq GPT-OSS 120B
        ↓
Structured Resume Analysis
        ↓
Actionable Recommendations
```

## 📋 Review Sections

The AI reviewer generates feedback covering:

1. **Overall Match**
2. **Matching Requirements**
3. **Requirements Not Mentioned**
4. **Experience Alignment**
5. **Resume Problems**
6. **Actionable Improvements**
7. **Suggested Keywords**
8. **Interview Preparation**
9. **Final Action Plan**

## 🛡️ Anti-Hallucination Approach

The reviewer is instructed **not to fabricate candidate information**.

For example, if a job description requires Docker but the resume does not mention Docker, the application should report:

> Not mentioned in the resume.

It should **not** assume that the candidate does or does not know Docker.

The application also avoids inventing:

* Skills
* Degrees
* Certifications
* Employers
* Projects
* Years of experience
* Achievements
* Technologies

## ⚠️ Limitations

* Scanned/image-only PDFs may not contain extractable text.
* Resume analysis depends on the quality of the provided resume and job description.
* Groq API rate limits may affect availability.
* AI-generated recommendations should be reviewed by the user before making changes to a resume.

## 📄 License

This project is intended for educational and portfolio purposes.
