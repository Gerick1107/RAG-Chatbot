import os
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

load_dotenv()

def smtp_configured():
    return all(os.getenv(k) for k in ("SMTP_HOST","SMTP_PORT","SMTP_USER","SMTP_PASS"))

def get_manager_email():
    return os.getenv("MANAGER_EMAIL","")

def send_email(to_email, subject, body, from_name=None):
    if not smtp_configured():
        return False, "SMTP is not configured. Add SMTP_* values to your .env file."
    host=os.getenv("SMTP_HOST")
    port=int(os.getenv("SMTP_PORT","587"))
    user=os.getenv("SMTP_USER")
    password=os.getenv("SMTP_PASS")
    if not to_email:
        return False, "No recipient address provided."
    msg=EmailMessage()
    msg["From"]=f"{from_name} <{user}>" if from_name else user
    msg["To"]=to_email
    msg["Subject"]=subject
    msg.set_content(body)
    try:
        if port==465:
            with smtplib.SMTP_SSL(host, port, timeout=20) as s:
                s.login(user, password); s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=20) as s:
                s.starttls(); s.login(user, password); s.send_message(msg)
        return True, f"Email sent to {to_email}."
    except Exception as e:
        return False, f"Failed to send email: {e}"
