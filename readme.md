# PaperDigest: AI Research Assistant

PaperDigest is a web application that uses Google's Gemini AI to summarize complex scientific papers and articles. It takes a URL (to a PDF or HTML page), extracts the core text, and provides a concise summary, key takeaways, and a simplified explanation of the methodology.

The application features real-time updates using WebSockets, a background worker for processing long-running tasks, and a complete user authentication system with job history.

## Key Features

* **AI-Powered Summarization**: Leverages Google Gemini to distill dense academic language into easy-to-understand summaries.
* **Multi-Format Support**: Can analyze content from both PDF documents and standard HTML web pages.
* **Asynchronous & Real-time**: Uses a background worker to handle analysis, with WebSockets pushing results to the user instantly upon completion. No more waiting on a loading screen!
* **User Authentication**: Secure user registration and login system.
* **Personal Job History**: Logged-in users have their entire analysis history saved to their account for future reference.
* **Robust & Reliable**: Built with modern best practices, including retries for network requests, safe parsing of AI responses, and a fallback OCR engine for scanned PDFs.

## Tech Stack

* **Backend**: FastAPI
* **Database**: SQLModel (on top of SQLAlchemy) with SQLite and `aiosqlite` for async support.
* **AI Model**: Google Gemini 1.5 Flash
* **Real-time Communication**: WebSockets
* **Authentication**: `fastapi-users` with JWT
* **PDF & Web Scraping**: `PyMuPDF`, `BeautifulSoup4`
* **OCR Fallback**: `pytesseract`
* **Testing**: `pytest` with `pytest-asyncio`

## Setup and Installation

Follow these steps to get the project running on your local machine.

### 1. Prerequisites

* Python 3.9+
* [Git](https://git-scm.com/downloads)
* Tesseract OCR Engine (see platform-specific installation instructions [here](https://tesseract-ocr.github.io/tessdoc/Installation.html)).

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd PaperDigest
```

### 3. Create and Activate Virtual Environment

It is highly recommended to use a virtual environment to manage project dependencies.

* **Create the environment:**
    ```bash
    python -m venv .venv
    ```
* **Activate the environment:**
    * On Windows:
        ```bash
        .venv\Scripts\activate
        ```
    * On macOS/Linux:
        ```bash
        source .venv/bin/activate
        ```

### 4. Install Dependencies

Install all the required Python libraries using the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 5. Set Up Environment Variables

The application needs your Google AI API key to function.

* Create a file named `.env` in the root of the project directory.
* Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
* Add the following line to your `.env` file, replacing the placeholder with your actual key:
    ```
    GOOGLE_API_KEY="your-google-api-key-goes-here"
    ```

## How to Run

The application consists of two main parts that must be run simultaneously: the web server and the background worker. You will need two separate terminal windows for this.

**Make sure your virtual environment is activated in both terminals.**

### Terminal 1: Start the FastAPI Web Server

This server handles API requests and WebSocket connections from the frontend.

```bash
uvicorn main:app --reload
```

The server will be running at `http://127.0.0.1:8000`.

### Terminal 2: Start the Background Worker

This script constantly checks the database for new jobs to process (downloading, analyzing, etc.).

```bash
python worker.py
```

### Accessing the Application

Once both the server and the worker are running, open a web browser and navigate to the `index.html` file in the project folder. You can now register, log in, and start analyzing articles.

## Running Tests

The project includes a comprehensive test suite to ensure all components are working correctly. The tests run against an isolated, in-memory database and will not affect your `jobs.db` file.

To run the tests, execute the following command in your terminal:

```bash
pytest -v
