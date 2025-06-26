# FILE: worker.py

import os
import fitz
import httpx
import google.generativeai as genai
import time
from sqlmodel import Field, Session, SQLModel, create_engine, select
from dotenv import load_dotenv
from typing import Optional


# --- Job Model Definition ---
# This must match the model in main.py
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
# The worker uses a synchronous engine because it works sequentially.
DATABASE_URL = f"sqlite:///{DATABASE_FILE}"
engine = create_engine(DATABASE_URL, echo=False)


# --- Core Logic ---
def get_text_from_url(url: str) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    response = httpx.get(url, timeout=30, headers=headers)
    response.raise_for_status()
    pdf_bytes = response.content
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return "".join(page.get_text() for page in doc)


def get_summary_from_ai(text: str) -> dict:
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # --- The Full, Engineered Prompt ---
    prompt = f"""
    You are an expert science journalist and communicator. Your primary task is to analyze the provided text from a scientific paper and generate a structured, easy-to-understand summary.

    Follow these steps precisely:

    1.  **Read the entire text** to understand its core concepts.
    2.  **Generate your response in the following strict XML format.** Do not include any text or explanations outside of these tags.

    <analysis>
      <title>Create a short, descriptive title for the paper (max 10 words) that captures its main topic, suitable for a general audience.</title>
      <summary>A single, concise paragraph (around 150 words) that explains the "what, why, and how" of the research at a high-school reading level. It should be easily understandable to someone outside the field. If new concepts need to be explained, generate rational explicative real-world analogies to help explain them to laymen.</summary>
      <methodology>In one or two simple sentences, describe the experiment or study design. For example, "The researchers analyzed survey data from 500 participants" or "They built a machine learning model to analyze images." Do not use technical jargon unless it is absolutely essential and explained.</methodology>
      <takeaways>
      ** In the takeways, do not use technical jargon. If complex terms need to be explained, define them simply as a laymen would understand. 
        <item>Extract the single most important finding or "so what?" of the paper.</item>
        <item>Extract the second most important finding.</item>
        <item>Extract a third key finding or an important limitation mentioned by the authors.</item>
        <item>Extract a reference to a potential real-world application of the findings, only if there is one.</item>
      </takeaways>
    </analysis>

    Here is the paper's text:
    <paper_text>
    {text[:900000]} 
    </paper_text>
    """
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
    print("--- Background Worker Started ---")
    while True:
        process_pending_job()
        time.sleep(5)


if __name__ == "__main__":
    SQLModel.metadata.create_all(engine)
    main()