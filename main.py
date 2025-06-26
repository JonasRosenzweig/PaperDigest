import os
import fitz
import requests
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from typing import Dict, List

# --- Setup ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# --- In-Memory Cache ---
analysis_cache: Dict[str, "AnalysisResponse"] = {}

app = FastAPI(
    title="Science Summarizer API (with Caching & Library)",
    description="An API to summarize scientific papers and view a library of cached results.",
)

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
    title: str  # NEW: Field for the paper's title
    summary: str
    takeaways: List[str]
    methodology: str
    cached: bool


# --- Core Logic Functions ---
def get_text_from_url(url: str) -> str:
    # This function remains the same.
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


def get_summary_from_ai(text: str) -> dict:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured.")

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # --- NEW: Updated prompt to include title extraction ---
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

        # --- NEW: Parse the new title tag ---
        title = message.split("<title>")[1].split("</title>")[0].strip()
        summary = message.split("<summary>")[1].split("</summary>")[0].strip()
        methodology = message.split("<methodology>")[1].split("</methodology>")[0].strip()
        takeaways_block = message.split("<takeaways>")[1].split("</takeaways>")[0]
        takeaways = [item.split("</item>")[0].strip() for item in takeaways_block.split("<item>")[1:]]
        return {"title": title, "summary": summary, "takeaways": takeaways, "methodology": methodology}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI analysis: {e}")


# --- API Endpoints ---
@app.post("/analyze", response_model=AnalysisResponse)
def analyze_paper(request: PaperRequest):
    request_url = str(request.url)
    print(f"Received request for URL: {request_url}")

    if request_url in analysis_cache:
        print("--- Cache Hit! Returning saved result. ---")
        cached_response = analysis_cache[request_url]
        cached_response.cached = True
        return cached_response

    print("--- Cache Miss. Starting new analysis. ---")
    paper_text = get_text_from_url(request_url)
    analysis_dict = get_summary_from_ai(paper_text)

    analysis_response = AnalysisResponse(**analysis_dict, cached=False)

    print("--- Saving new result to cache. ---")
    analysis_cache[request_url] = analysis_response

    print("Analysis complete.")
    return analysis_response


# --- NEW: Endpoint to get the library of cached items ---
@app.get("/library")
def get_library():
    return analysis_cache