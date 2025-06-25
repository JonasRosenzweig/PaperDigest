import os
import fitz  # PyMuPDF
import requests
import google.generativeai as genai  # <-- CHANGED
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load the secret API key from the .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # <-- CHANGED

# --- FastAPI App Setup ---
app = FastAPI(
    title="Science Summarizer API (Gemini Edition)",  # <-- CHANGED
    description="An API to summarize scientific papers from a URL using Google Gemini.",
)

# This is crucial for allowing our frontend to talk to our backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models ---
class PaperRequest(BaseModel):
    url: HttpUrl


class AnalysisResponse(BaseModel):
    summary: str
    takeaways: list[str]
    methodology: str


# --- Core Logic Functions ---

def get_text_from_url(url: str) -> str:
    """Downloads a PDF from a URL and extracts all text. (No changes here)"""
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

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
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the PDF: {e}")


def get_summary_from_ai(text: str) -> AnalysisResponse:
    """Sends text to Google's Gemini AI for analysis and returns a structured response."""
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured.")

    try:
        # --- Configure the Gemini client --- (CHANGED)
        genai.configure(api_key=GOOGLE_API_KEY)

        # Use the fast and powerful Gemini 1.5 Flash model
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # The prompt is the same, but Gemini can handle a much larger context
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
        {text[:900000]} 
        </paper_text>
        """  # Gemini 1.5 Flash has a huge 1M token context window!

        # --- Generate content using the Gemini model --- (CHANGED)
        response = model.generate_content(prompt)
        message = response.text

        # --- Simple XML Parsing (No changes here) ---
        summary = message.split("<summary>")[1].split("</summary>")[0].strip()
        methodology = message.split("<methodology>")[1].split("</methodology>")[0].strip()
        takeaways_block = message.split("<takeaways>")[1].split("</takeaways>")[0]
        takeaways = [item.split("</item>")[0].strip() for item in takeaways_block.split("<item>")[1:]]

        return AnalysisResponse(summary=summary, takeaways=takeaways, methodology=methodology)

    except Exception as e:
        # This will catch errors from the Gemini API as well
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI analysis: {e}")


# --- API Endpoint (No changes here) ---

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