import os
import fitz  # PyMuPDF
import requests
import anthropic
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load the secret API key from the .env file
load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# --- FastAPI App Setup ---
app = FastAPI(
    title="Science Summarizer API",
    description="An API to summarize scientific papers from a URL.",
)

# This is crucial for allowing our frontend to talk to our backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins for simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class PaperRequest(BaseModel):
    url: HttpUrl  # FastAPI will automatically validate that this is a real URL


class AnalysisResponse(BaseModel):
    summary: str
    takeaways: list[str]
    methodology: str


# --- Core Logic Functions ---

def get_text_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts all text."""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()  # Raises an error for bad responses (4xx or 5xx)

        # Ensure the content is a PDF
        if "application/pdf" not in response.headers.get("Content-Type", ""):
            raise HTTPException(status_code=400, detail="URL does not point to a PDF file.")

        pdf_bytes = response.content
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)

        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        return full_text
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to download the file: {e}")
    except Exception as e:
        # Catches PyMuPDF errors or other unexpected issues
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the PDF: {e}")


def get_summary_from_ai(text: str) -> AnalysisResponse:
    """Sends text to Anthropic's AI for analysis and returns a structured response."""
    if not ANTHROPIC_API_KEY:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured.")

    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

        # We use a powerful Haiku model which is fast and cheap
        # For higher quality, you could swap this for 'claude-3-sonnet-20240229'
        model_name = "claude-3-haiku-20240307"

        # The "Master Prompt" that instructs the AI
        prompt = f"""
        You are an expert science communicator. Your task is to analyze the following scientific paper and provide a clear, structured summary for a non-expert audience. Please provide your response in the following XML format, and do not include any other text before or after the XML tags.

        <analysis>
          <summary>A one-paragraph summary of the paper's main goal, methods, and conclusion.</summary>
          <takeaways>
            <item>Key takeaway number one.</item>
            <item>Key takeaway number two.</item>
            <item>Key takeaway number three.</item>
          </takeaways>
          <methodology>A simple explanation of how the researchers conducted their study. Avoid jargon.</methodology>
        </analysis>

        Here is the paper's text:
        <paper_text>
        {text[:20000]} 
        </paper_text>
        """  # We truncate the text to avoid exceeding token limits for the MVP

        message = client.messages.create(
            model=model_name,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": prompt}
            ]
        ).content[0].text

        # --- Simple XML Parsing ---
        # A more robust solution would use an XML library, but this is fine for the MVP
        summary = message.split("<summary>")[1].split("</summary>")[0].strip()
        methodology = message.split("<methodology>")[1].split("</methodology>")[0].strip()
        takeaways_block = message.split("<takeaways>")[1].split("</takeaways>")[0]
        takeaways = [item.split("</item>")[0].strip() for item in takeaways_block.split("<item>")[1:]]

        return AnalysisResponse(summary=summary, takeaways=takeaways, methodology=methodology)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI analysis: {e}")


# --- API Endpoint ---

@app.post("/analyze", response_model=AnalysisResponse)
def analyze_paper(request: PaperRequest):
    """
    The main endpoint that receives a URL, orchestrates the analysis,
    and returns the structured summary.
    """
    print(f"Analyzing URL: {request.url}")
    paper_text = get_text_from_url(str(request.url))
    analysis = get_summary_from_ai(paper_text)
    print("Analysis complete.")
    return analysis