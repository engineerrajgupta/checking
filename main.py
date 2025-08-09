# main.py (FINAL - BULLETPROOF)

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Import our new functions
from logic import process_document_and_questions, setup_database
from db import init_db_pool, close_db_pool

# --- Lifespan Event ---
# This will create the pool on startup and close it on shutdown.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    init_db_pool()
    setup_database()
    yield
    print("Application shutdown...")
    close_db_pool()

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query-Retrieval API",
    description="An API that uses a persistent, connection-pooled database cache.",
    version="14.0.0-bulletproof",
    lifespan=lifespan
)

# --- CORS Middleware ---
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- Pydantic Models ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

# --- API Endpoint ---
@app.post("/hackrx/run")
def run_submission(request_data: QueryRequest) -> Dict[str, Any]:
    try:
        results = process_document_and_questions(
            pdf_url=request_data.documents, 
            questions=request_data.questions
        )
        if "error" in results: raise HTTPException(status_code=400, detail=results["error"])
        return results
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "API is running"}
