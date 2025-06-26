import os
import fitz
import httpx  # Replaces 'requests'
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
    title="Async Science Summarizer API",
    description="An asynchronous API to summarize scientific papers.",
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
    title: str
    summary: str
    takeaways: List[str]
    methodology: str
    cached: bool


# --- Core Logic Functions (Now Asynchronous!) ---
async def get_text_from_url(url: str) -> str:
    # Use an async HTTP client that doesn't block the server
    async with httpx.AsyncClient() as client:
        try:
            # Add a standard browser User-Agent to prevent getting blocked
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            response = await client.get(url, timeout=30, headers=headers)
            response.raise_for_status()
        except httpx.RequestError as e:
            raise HTTPException(status_code=400, detail=f"Failed to download the file: {e}")

    if "application/pdf" not in response.headers.get("Content-Type", ""):
        raise HTTPException(status_code=400, detail="URL does not point to a PDF file.")

    pdf_bytes = await response.aread()

    try:
        # PyMuPDF is CPU-bound, so it runs fine inside an async function.
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)
        if not full_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")
        return full_text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the PDF: {e}")


async def get_summary_from_ai(text: str) -> dict:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY not configured.")

    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        # This is the prompt from the code you uploaded
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
        # Use the async version of the Gemini library method
        response = await model.generate_content_async(prompt)
        message = response.text

        title = message.split("<title>")[1].split("</title>")[0].strip()
        summary = message.split("<summary>")[1].split("</summary>")[0].strip()
        methodology = message.split("<methodology>")[1].split("</methodology>")[0].strip()
        takeaways_block = message.split("<takeaways>")[1].split("</takeaways>")[0]
        takeaways = [item.split("</item>")[0].strip() for item in takeaways_block.split("<item>")[1:]]
        return {"title": title, "summary": summary, "takeaways": takeaways, "methodology": methodology}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred with the AI analysis: {e}")


# --- API Endpoints (Now Asynchronous!) ---
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_paper(request: PaperRequest):
    request_url = str(request.url)
    print(f"Received request for URL: {request_url}")

    if request_url in analysis_cache:
        print("--- Cache Hit! Returning saved result. ---")
        cached_response = analysis_cache[request_url]
        # Create a new response object to set the cached flag
        return AnalysisResponse(**cached_response.dict(exclude={'cached'}), cached=True)

    print("--- Cache Miss. Starting new analysis. ---")
    paper_text = await get_text_from_url(request_url)
    analysis_dict = await get_summary_from_ai(paper_text)

    analysis_response = AnalysisResponse(**analysis_dict, cached=False)

    print("--- Saving new result to cache. ---")
    analysis_cache[request_url] = analysis_response

    print("Analysis complete.")
    return analysis_response


@app.get("/library")
async def get_library():
    # This endpoint is simple and doesn't do I/O, so it can remain async
    # without any `await` calls.
    return analysis_cache