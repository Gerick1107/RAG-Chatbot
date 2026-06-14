"""
HR Assistant — a minimal multi-module Streamlit app powered by Groq.

Modules (switchable from the sidebar):
  1. HR FAQ / Policy Chatbot   — RAG over company_docs/ + onboarding generators
  2. Resume Screening          — score resumes against a job description
  3. Leave Request Assistant   — draft & email a leave request from plain English
  4. Interview Questions       — HR generates questions to ask a candidate (+ resume-specific)

Run with:  streamlit run app.py
"""
import os
import json
import streamlit as st

from rag_core import (
    MODEL, TOP_K, build_company_index, list_company_docs, retrieve,
    chat, answer_with_context, read_uploaded,
)
from email_utils import send_email, smtp_configured, get_manager_email

st.set_page_config(page_title="HR Assistant", page_icon="🧑‍💼", layout="wide")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY not found. Add it to the .env file and restart.")
    st.stop()


# ====================================================================== utils
def parse_json(text):
    """Best-effort extraction of a JSON object/array from an LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:]
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start, end = text.find(open_c), text.rfind(close_c)
        if start != -1 and end != -1 and end > start:
            snippet = text[start:end + 1]
            # strict=False tolerates raw newlines inside strings, which LLMs
            # routinely emit in multi-line fields like an email body.
            try:
                return json.loads(snippet, strict=False)
            except json.JSONDecodeError:
                continue
    return None


# ============================================================ 1. HR FAQ / RAG
def module_hr_faq():
    st.header("💬 HR FAQ / Policy")
    st.caption("Ask about company policies, or generate onboarding documents for new joiners.")

    chunks, embeddings, sources = build_company_index()
    docs = list_company_docs()
    if not docs:
        st.warning("No PDFs found in `company_docs/`. Drop in policy PDFs and refresh.")
        return
    st.info("Indexed documents: " + ", ".join(f"`{d}`" for d in docs))

    tab_faq, tab_gen = st.tabs(["HR FAQ / Policy", "Generate"])

    with tab_faq:
        st.session_state.setdefault("faq_messages", [])
        for msg in st.session_state.faq_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        examples = ["How many casual leaves do I have?",
                    "What is the WFH policy?",
                    "What documents are needed for reimbursement?"]
        st.caption("Try: " + "  •  ".join(examples))

        if q := st.chat_input("Ask about HR policies..."):
            st.session_state.faq_messages.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    hits, hit_sources = retrieve(q, chunks, embeddings, sources)
                    answer = answer_with_context(q, hits)
                st.markdown(answer)
                with st.expander("Retrieved context"):
                    for i, (chunk, src) in enumerate(zip(hits, hit_sources), 1):
                        st.markdown(f"**{i}. _{src}_** — {chunk[:300]}...")
            st.session_state.faq_messages.append({"role": "assistant", "content": answer})

    with tab_gen:
        st.caption("Generate onboarding documents for a new joiner.")
        role = st.text_input("New joiner's role", value="Software Engineer")
        what = st.radio("What would you like to generate?",
                        ["First-week checklist", "Joining email draft", "Orientation schedule"],
                        horizontal=True)
        if st.button("Generate", type="primary"):
            hits, _ = retrieve(what + " " + role, chunks, embeddings, sources, top_k=TOP_K)
            context = "\n\n".join(hits)
            instructions = {
                "First-week checklist": "Create a clear, friendly day-by-day first-week "
                    "checklist (Day 1 to Day 5) as markdown with checkboxes.",
                "Joining email draft": "Write a warm welcome/joining email from HR to the "
                    "new joiner with key Day-1 details.",
                "Orientation schedule": "Create a Day-1 orientation schedule as a markdown "
                    "table with time slots and sessions.",
            }[what]
            prompt = (
                f"You are an HR assistant. {instructions} The new joiner's role is '{role}'. "
                "Use the company context below where relevant; if a detail isn't there, use "
                "sensible generic defaults.\n\n"
                f"COMPANY CONTEXT:\n{context or '(none available)'}"
            )
            with st.spinner("Generating..."):
                out = chat([{"role": "user", "content": prompt}], temperature=0.4, max_tokens=900)
            st.markdown(out)


# ====================================================== 2. Resume Screening
def screen_resume(jd, resume_text):
    prompt = (
        "You are a technical recruiter. Compare the candidate's resume against the "
        "job description and respond with ONLY a JSON object of this exact shape:\n"
        "{\n"
        '  "candidate_name": string,\n'
        '  "match_score": integer 0-100,\n'
        '  "skills": [up to 8 short skill strings],\n'
        '  "strengths": [2-4 short strings],\n'
        '  "gaps": [1-4 short strings],\n'
        '  "interview_questions": [3-5 question strings]\n'
        "}\n\n"
        f"JOB DESCRIPTION:\n{jd}\n\nRESUME:\n{resume_text}"
    )
    raw = chat([{"role": "user", "content": prompt}], temperature=0.2, max_tokens=900)
    return parse_json(raw)


def module_resume_screening():
    st.header("📋 Recruitment / Resume Screening")
    st.caption("Paste a job description, upload resumes, and get a scored comparison.")

    jd = st.text_area("Job Description", height=180,
                      placeholder="Paste the full job description here...")
    files = st.file_uploader("Upload resume PDFs", type=["pdf", "txt"],
                             accept_multiple_files=True)

    if st.button("Screen resumes", type="primary"):
        if not jd.strip():
            st.warning("Please paste a job description first.")
            return
        if not files:
            st.warning("Please upload at least one resume.")
            return

        results = []
        progress = st.progress(0.0)
        for n, f in enumerate(files, 1):
            with st.spinner(f"Analysing {f.name}..."):
                text = read_uploaded(f)
                data = screen_resume(jd, text)
            if data:
                data["_file"] = f.name
                results.append(data)
            else:
                st.error(f"Couldn't parse the analysis for {f.name}. Try again.")
            progress.progress(n / len(files))
        progress.empty()
        st.session_state.screen_results = results

    results = st.session_state.get("screen_results")
    if not results:
        return

    results = sorted(results, key=lambda r: r.get("match_score", 0), reverse=True)

    st.subheader("Comparison")
    table = [{
        "Candidate": r.get("candidate_name") or r.get("_file"),
        "Match %": r.get("match_score", 0),
        "Top skills": ", ".join(r.get("skills", [])[:5]),
        "File": r.get("_file"),
    } for r in results]
    st.dataframe(table, use_container_width=True,
                 column_config={"Match %": st.column_config.ProgressColumn(
                     "Match %", min_value=0, max_value=100, format="%d%%")})

    for r in results:
        name = r.get("candidate_name") or r.get("_file")
        with st.expander(f"{name} — {r.get('match_score', 0)}% match"):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**✅ Strengths**")
                for s in r.get("strengths", []):
                    st.markdown(f"- {s}")
                st.markdown("**🧩 Skills**")
                st.markdown(", ".join(r.get("skills", [])) or "_none extracted_")
            with c2:
                st.markdown("**⚠️ Gaps**")
                for g in r.get("gaps", []):
                    st.markdown(f"- {g}")
            st.markdown("**❓ Suggested interview questions**")
            for i, qn in enumerate(r.get("interview_questions", []), 1):
                st.markdown(f"{i}. {qn}")


# ===================================================== 3. Leave Request
def draft_leave(request_text):
    prompt = (
        "An employee wrote a casual leave request in plain language. Extract the "
        "details and draft a formal leave-request email. Respond with ONLY a JSON "
        "object of this shape:\n"
        "{\n"
        '  "start_date": string (e.g. \\"June 20, 2026\\" or \\"unspecified\\"),\n'
        '  "end_date": string,\n'
        '  "reason": short string,\n'
        '  "subject": email subject line,\n'
        '  "body": full formal email body addressed to the manager, signed off as the employee\n'
        "}\n\n"
        f"EMPLOYEE REQUEST:\n{request_text}"
    )
    raw = chat([{"role": "user", "content": prompt}], temperature=0.3, max_tokens=700)
    return parse_json(raw)


def module_leave_request():
    st.header("📝 Leave Request Assistant")
    st.caption("Describe your leave in plain English — we'll draft and (optionally) email it.")

    req = st.text_area("Your leave request", height=110,
                       placeholder="e.g. I want leave from June 20 to June 25 for a family function.")

    if st.button("Draft request", type="primary"):
        if not req.strip():
            st.warning("Please describe your leave request first.")
            return
        with st.spinner("Drafting..."):
            data = draft_leave(req)
        if not data:
            st.error("Couldn't draft the request. Please try rephrasing.")
            return
        st.session_state.leave_draft = data

    data = st.session_state.get("leave_draft")
    if not data:
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("From", data.get("start_date", "—"))
    c2.metric("To", data.get("end_date", "—"))
    c3.metric("Reason", data.get("reason", "—"))

    subject = st.text_input("Subject", value=data.get("subject", "Leave Request"))
    body = st.text_area("Email body", value=data.get("body", ""), height=260)

    st.divider()
    default_to = get_manager_email()
    to_email = st.text_input("Send to (manager or any recipient)",
                             value=default_to, placeholder="recipient@company.com")
    sender_name = st.text_input("Your name (for the From field)", value="")

    if not smtp_configured():
        st.info("📭 SMTP isn't configured, so emails can't be sent yet. You can "
                "still copy the draft above. To enable sending, add `SMTP_HOST`, "
                "`SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, and `MANAGER_EMAIL` to your "
                "`.env` file (see the README).")

    if st.button("Send email", disabled=not smtp_configured()):
        ok, message = send_email(to_email, subject, body, from_name=sender_name or None)
        (st.success if ok else st.error)(message)


# ===================================================== 4. Interview Questions
def module_interview_questions():
    st.header("🎯 Interview Questions")
    st.caption("Generate questions for HR to ask a candidate — by role, or based on their resume.")

    role = st.text_input("Job role", placeholder="e.g. Python Backend Developer")
    c1, c2 = st.columns(2)
    level = c1.selectbox("Experience level", ["Junior", "Mid", "Senior"])
    count = c2.slider("General questions", 5, 15, 8)

    resume_file = st.file_uploader("Upload candidate's resume (optional — adds resume-specific questions)",
                                   type=["pdf", "txt"])

    if st.button("Generate questions", type="primary"):
        if not role.strip():
            st.warning("Please enter a role.")
            return

        general_prompt = (
            f"Generate {count} interview questions for HR to ask a {level}-level '{role}' "
            "candidate. Format as a numbered markdown list grouped under "
            "**Technical**, **Problem-solving**, and **Behavioural** headings."
        )
        with st.spinner("Generating general questions..."):
            general_out = chat([{"role": "user", "content": general_prompt}],
                               temperature=0.5, max_tokens=900)
        st.markdown(general_out)

        if resume_file:
            resume_text = read_uploaded(resume_file)
            resume_prompt = (
                f"You are an interviewer preparing to interview a {level}-level '{role}' candidate. "
                "Based on the resume below, generate 5 targeted interview questions that probe "
                "specific experiences, projects, or skills mentioned. "
                "Format as a numbered markdown list under a **Resume-specific** heading.\n\n"
                f"RESUME:\n{resume_text}"
            )
            with st.spinner("Generating resume-specific questions..."):
                resume_out = chat([{"role": "user", "content": resume_prompt}],
                                  temperature=0.4, max_tokens=600)
            st.markdown(resume_out)


# ============================================================== sidebar / nav
MODULES = {
    "💬 HR FAQ / Policy": module_hr_faq,
    "📋 Resume Screening": module_resume_screening,
    "📝 Leave Request": module_leave_request,
    "🎯 Interview Questions": module_interview_questions,
}

with st.sidebar:
    st.title("🧑‍💼 HR Assistant")
    st.caption(f"Powered by {MODEL} via Groq")
    choice = st.radio("Module", list(MODULES.keys()), label_visibility="collapsed")
    st.divider()
    st.caption("Company docs live in `company_docs/`. "
               "Drop in PDFs and the HR FAQ module picks them up.")
    if smtp_configured():
        st.success("SMTP configured ✓")
    else:
        st.warning("SMTP not configured")

MODULES[choice]()
