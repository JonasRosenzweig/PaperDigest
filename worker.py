# FILE: worker.py

import os
import fitz
import httpx
import google.generativeai as genai
import time
from sqlmodel import Field, Session, SQLModel, create_engine, select
from dotenv import load_dotenv
from typing import Optional
from bs4 import BeautifulSoup


# --- Job Model Definition ---
class Job(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    url: str
    status: str
    title: Optional[str] = None
    summary: Optional[str] = None
    takeaways: Optional[str] = None
    methodology: Optional[str] = None
    error_message: Optional[str] = None


# --- Setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
DATABASE_FILE = "jobs.db"
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"
engine = create_engine(DATABASE_URL, echo=False)

# NEW: Read the prompt from the text file
try:
    with open("prompt.txt", "r") as f:
        PROMPT_TEMPLATE = f.read()
except FileNotFoundError:
    print("ERROR: prompt.txt not found. Please create it.")
    PROMPT_TEMPLATE = ""  # Set a default to avoid crashing


# --- Core Logic ---
def get_text_from_url(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    print(f"Fetching content from {url}...")
    response = httpx.get(url, timeout=30, headers=headers, follow_redirects=True)
    response.raise_for_status()
    content_type = response.headers.get('content-type', '').lower()
    text = ""

    if 'application/pdf' in content_type:
        print("PDF detected. Extracting text with PyMuPDF.")
        pdf_bytes = response.content
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
    elif 'text/html' in content_type:
        print("HTML detected. Extracting text with BeautifulSoup.")
        soup = BeautifulSoup(response.text, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = ' '.join(t.strip() for t in soup.stripped_strings)
    else:
        raise Exception(f"Unsupported content type: {content_type}")

    if not text:
        raise Exception("Could not extract any text from the URL.")
    return text


def get_summary_from_ai(text: str) -> dict:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # NEW: Format the prompt using the template we loaded from the file
    prompt = PROMPT_TEMPLATE.format(text=text[:900000])

    response = model.generate_content(prompt)
    message = response.text

    title = message.split("<title>")[1].split("</title>")[0].strip()
    summary = message.split("<summary>")[1].split("</summary>")[0].strip()
    methodology = message.split("<methodology>")[1].split("</methodology>")[0].strip()
    takeaways_block = message.split("<takeaways>")[1].split("</takeaways>")[0]
    takeaways = [item.split("</item>")[0].strip() for item in takeaways_block.split("<item>")[1:]]
    return {"title": title, "summary": summary, "takeaways": takeaways, "methodology": methodology}


# --- Main Worker Loop ---
def process_pending_job():
    with Session(engine) as session:
        job_to_process = session.exec(select(Job).where(Job.status == "PENDING")).first()
        if not job_to_process:
            print("No pending jobs found. Waiting...")
            return

        print(f"Processing job ID: {job_to_process.id} for URL: {job_to_process.url}")
        job_to_process.status = "PROCESSING"
        session.add(job_to_process)
        session.commit()
        session.refresh(job_to_process)

        try:
            text = get_text_from_url(job_to_process.url)
            ai_result = get_summary_from_ai(text)

            job_to_process.status = "COMPLETED"
            job_to_process.title = ai_result["title"]
            job_to_process.summary = ai_result["summary"]
            job_to_process.methodology = ai_result["methodology"]
            job_to_process.takeaways = "|||".join(ai_result["takeaways"])
        except Exception as e:
            print(f"ERROR processing job {job_to_process.id}: {e}")
            job_to_process.status = "FAILED"
            job_to_process.error_message = str(e)

        session.add(job_to_process)
        session.commit()
        print(f"Job ID: {job_to_process.id} finished with status: {job_to_process.status}")


def main():
    print("--- Background Worker Started (with HTML support & external prompt) ---")
    while True:
        process_pending_job()
        time.sleep(5)


if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)
    main()