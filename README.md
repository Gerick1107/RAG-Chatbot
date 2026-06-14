# 🧑‍💼 HR Assistant

A minimal, clean multi-module HR assistant built with **Streamlit** and the
**Groq API** (`llama-3.1-8b-instant`). It started life as a single-file RAG
chatbot and now bundles four HR modules behind one sidebar — no auth, no
database, no Docker. Just `streamlit run app.py`.

## Modules

| Module | What it does |
|--------|--------------|
| 💬 **HR FAQ / Policy** | RAG over your `company_docs/` PDFs. Ask "How many casual leaves do I have?", "What is the WFH policy?", "What documents are needed for reimbursement?" Also has a **Generate** tab for onboarding documents: first-week checklist, joining email, or orientation schedule. |
| 📋 **Resume Screening** | Paste a job description, upload one or more resume PDFs. For each: extracted skills, a match score, strengths, gaps, and suggested interview questions — in a sortable comparison table. |
| 📝 **Leave Request** | Type a request in plain English ("I want leave from June 20 to June 25"). It extracts the dates/reason, drafts a formal email, and can send it to any recipient email you specify. |
| 🎯 **Interview Questions** | HR generates questions to ask a candidate — by role and experience level, plus optional resume upload for 5 targeted resume-specific questions. |

## How the RAG pipeline works

1. **Chunking** — each PDF in `company_docs/` is split into ~200-word chunks (40-word overlap).
2. **Embedding** — chunks become vectors via `sentence-transformers` (`all-MiniLM-L6-v2`), normalized and cached in memory.
3. **Retrieval** — your question is embedded the same way; the top 4 chunks are found by cosine similarity.
4. **Generation** — those chunks are passed as grounding context to the Groq LLM.

## Setup

1. **Install dependencies** (Python 3.10+):

   ```bash
   pip install -r requirements.txt
   ```

2. **Add your Groq API key** — copy `.env.example` to `.env` and fill it in:

   ```
   GROQ_API_KEY=gsk_your_actual_key_here
   ```

   Get a free key at https://console.groq.com/keys

3. **(Optional) Generate sample data** — creates example policy PDFs and resumes:

   ```bash
   python generate_sample_pdfs.py
   ```

   This populates `company_docs/` (HR policy, reimbursement, onboarding,
   handbook) and `sample_resumes/` (Python developer, data analyst, full-stack).

4. **Run the app:**

   ```bash
   streamlit run app.py
   ```

   Opens at `http://localhost:8501`.

## Adding your own company PDFs

Drop any `.pdf` files into the **`company_docs/`** folder. The HR FAQ and
Onboarding modules index every PDF in that folder automatically on startup.
After adding or changing files, restart the app (or press **R** to rerun) so the
index rebuilds.

## Configuring SMTP (for the Leave Request "Send email" button)

Email sending is **optional** — without it you can still draft and copy the
leave request. To enable sending, add these to your `.env`:

```
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your_email@gmail.com
SMTP_PASS=your_app_password_here
MANAGER_EMAIL=manager@company.com
```

**Gmail setup (most common):**

1. Enable **2-Step Verification** on your Google account.
2. Go to https://myaccount.google.com/apppasswords and create an **App Password**
   (a 16-character code). Use that as `SMTP_PASS` — *not* your normal Gmail password.
3. Set `SMTP_USER` to your Gmail address and `MANAGER_EMAIL` to where requests should go.

**Other providers:**

| Provider | SMTP_HOST | SMTP_PORT |
|----------|-----------|-----------|
| Gmail | `smtp.gmail.com` | `587` |
| Outlook / Office 365 | `smtp.office365.com` | `587` |
| Yahoo | `smtp.mail.yahoo.com` | `587` |

Port `587` uses STARTTLS (the default); port `465` uses SSL — both are handled
automatically. The sidebar shows whether SMTP is configured.

## Files

| File | Purpose |
|------|---------|
| `app.py` | The Streamlit app — sidebar nav + the four modules |
| `rag_core.py` | Groq client, embeddings, the in-memory vector store, LLM helpers |
| `email_utils.py` | SMTP send helper (reads credentials from `.env`) |
| `generate_sample_pdfs.py` | One-off script to create sample `company_docs/` & `sample_resumes/` |
| `company_docs/` | Policy PDFs indexed for the FAQ & Onboarding modules |
| `sample_resumes/` | Example resumes for testing Resume Screening |
| `.env` / `.env.example` | Secrets (gitignored) and a template |

> **Note:** the first run downloads the embedding model (~90 MB), so it may take a minute.
