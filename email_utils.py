"""
Minimal SMTP helper. All credentials are read from environment variables so no
secrets live in code:

    SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, MANAGER_EMAIL

Most providers (Gmail, Outlook, etc.) use STARTTLS on port 587, which is the
default behaviour here.
"""
import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

load_msg = "SMTP is not configured. Add SMTP_* values to your .env file."


def smtp_configured():
    return all(os.getenv(k) for k in ("SMTP_HOST", "SMTP_PORT", "SMTP_USER", "SMTP_PASS"))


def get_manager_email():
    return os.getenv("MANAGER_EMAIL", "")


def send_email(to_email, subject, body, from_name=None):
    """
    Send a plain-text email via SMTP. Returns (ok: bool, message: str).
    Never raises — failures are returned so the UI can show them cleanly.
    """
    if not smtp_configured():
        return False, load_msg

    host = os.getenv("SMTP_HOST")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER")
    password = os.getenv("SMTP_PASS")

    if not to_email:
        return False, "No recipient address. Set MANAGER_EMAIL in .env or enter one."

    msg = EmailMessage()
    msg["From"] = f"{from_name} <{user}>" if from_name else user
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as server:
                server.login(user, password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as server:
                server.starttls()
                server.login(user, password)
                server.send_message(msg)
        return True, f"Email sent to {to_email}."
    except Exception as e:
        return False, f"Failed to send email: {e}"
