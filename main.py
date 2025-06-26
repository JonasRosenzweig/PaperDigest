import asyncio
import os
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl
from typing import Optional, List
from sqlmodel import Field, Session, SQLModel, create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

# --- Database Setup ---
DATABASE_FILE = "jobs.db"
DATABASE_URL = f"sqlite+aiosqlite:///{DATABASE_FILE}"

engine = create_async_engine(DATABASE_URL, echo=False)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


# --- Database Models ---
class Job(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    url: str
    status: str = "PENDING"
    title: Optional[str] = None
    summary: Optional[str] = None
    takeaways: Optional[str] = None
    methodology: Optional[str] = None
    error_message: Optional[str] = None


# This runs on server startup to create the DB table if it doesn't exist.
async def create_db_and_tables():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)


# --- FastAPI App Setup ---
app = FastAPI(
    title="Real-time Science Summarizer API",
    on_startup=[create_db_and_tables],
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)


class PaperRequest(BaseModel):
    url: HttpUrl


# --- API Endpoints ---
@app.post("/start-analysis", response_model=Job)
async def start_analysis_job(request: PaperRequest):
    """Instantly creates a job in the database and returns it."""
    async with AsyncSessionLocal() as session:
        new_job = Job(url=str(request.url))
        session.add(new_job)
        await session.commit()
        await session.refresh(new_job)
        return new_job


@app.get("/library", response_model=List[Job])
async def get_library():
    """Returns all successfully completed jobs to populate the library."""
    async with AsyncSessionLocal() as session:
        statement = select(Job).where(Job.status == "COMPLETED")
        results = await session.execute(statement)
        completed_jobs = results.scalars().all()
        return completed_jobs


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: int):
    """Handles the real-time connection for a specific job."""
    await websocket.accept()
    try:
        while True:
            async with AsyncSessionLocal() as session:
                job = await session.get(Job, job_id)

            if not job:
                await websocket.send_json({"status": "ERROR", "message": "Job not found."})
                break

            if job.status in ["COMPLETED", "FAILED"]:
                await websocket.send_json(job.dict())
                break
            else:
                await websocket.send_json({"status": job.status})
                await asyncio.sleep(3)
    except WebSocketDisconnect:
        print(f"Client disconnected from job {job_id}")
    finally:
        await websocket.close()