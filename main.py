# FILE: main.py
# This version adds print statements to debug the API key loading process.

import os
import fitz
import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# --- Setup ---
print("--- Loading environment variables from .env file... ---")
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- NEW: Debugging Print Statement ---
if GOOGLE_API_KEY:
    print("SUCCESS: GOOGLE_API_KEY loaded successfully.")
    # For security, let's only print the first few characters
    print(f"API Key starts with: {GOOGLE_API_KEY[:4]}...")
else:
    print("ERROR: GOOGLE_API_KEY not found in environment. Please check your .env file.")
    # We can exit here if the key is missing, but for now we'll let it fail later.


app = FastAPI(
    title="Science Summarizer API (Gemini Edition)",
    description="An API to summarize scientific papers from a URL using Google Gemini.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Pydantic Models (No changes) ---
class PaperRequest(BaseModel):
    url: HttpUrl

class AnalysisResponse(BaseModel):
    summary: str
    takeaways: list[str]
    methodology: str


# --- Core Logic Functions (No changes) ---
def get_text_from_url(url: str) -> str:
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
    if not GOOGLE_API_KEY:
        # This will now be caught at startup, but we keep it as a safeguard.
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured.")

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        prompt = f"""
        You are an expert science journalist and communicator, tasked with distilling a complex academic paper into a clear, accessible, and accurate summary for a curious but non-technical audience (e.g., an undergraduate student).

        Your task is to follow a two-step process:

        **Step 1: Internal Analysis (Your Scratchpad)**
        First, think step-by-step. Read the entire provided text and, inside a `<scratchpad>` XML tag, identify and list the following core components:
        - **Problem:** What is the core research question or the problem the authors are trying to solve?
        - **Methodology:** How did the researchers conduct their study? (e.g., "analyzed survey data," "conducted a randomized trial," "built a new type of neural network").
        - **Results:** What were the key findings or observations? List the most important data points or outcomes.
        - **Conclusion:** What is the main conclusion or interpretation the authors draw from their results?

        **Step 2: Final Output Generation**
        After completing your internal analysis, and using ONLY the information you gathered in the scratchpad, generate the final public response in the following strict XML format. Do not include any other text or explanation outside of these XML tags.

        <analysis>
          <summary>A single, concise paragraph (around 150 words) that explains the "what, why, and how" of the research at a high-school reading level. It should be easily understandable to someone outside the field.</summary>
          <takeaways>
            <item>Extract the single most important finding or "so what?" of the paper.</item>
            <item>Extract the second most important finding.</item>
            <item>Extract a third key finding or an important limitation mentioned by the authors.</item>
          </takeaways>
          <methodology>In one or two simple sentences, describe the experiment or study design. For example, "The researchers analyzed survey data from 500 participants" or "They built a machine learning model to analyze images." Do not use technical jargon unless it is absolutely essential and explained.</methodology>
        </analysis>

        Do not invent or infer any information not present in the provided text.

        Here is the paper's text:
        <paper_text>
        {text[:900000]} 
        </paper_text>
        """
        response = model.generate_content(prompt)
        message = response.text
        summary = message.split("<summary>")[1].split("</summary>")[0].strip()
        methodology = message.split("<methodology>")[1].split("</methodology>")[0].strip()
        takeaways_block = message.split("<takeaways>")[1].split("</takeaways>")[0]
        takeaways = [item.split("</item>")[0].strip() for item in takeaways_block.split("<item>")[1:]]
        return AnalysisResponse(summary=summary, takeaways=takeaways, methodology=methodology)
    except Exception as e:
        print(f"--- ERROR FROM GOOGLE AI --- \n{e}\n--------------------------")
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI analysis: {e}")


# --- API Endpoint (No changes) ---
@app.post("/analyze", response_model=AnalysisResponse)
def analyze_paper(request: PaperRequest):
    print(f"Analyzing URL: {request.url}")
    paper_text = get_text_from_url(str(request.url))
    analysis = get_summary_from_ai(paper_text)
    print("Analysis complete.")
    return analysis