# main.py
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from db import init_db_pool, close_db_pool
import logic

class QueryRequest(BaseModel):
    documents: str   # pdf URL
    questions: List[str]

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    init_db_pool()
    # ensure DB table is ready
    try:
        logic.ensure_table_exists()
    except Exception as e:
        print(f"setup_database error: {e}")
    yield
    print("Application shutdown...")
    close_db_pool()

app = FastAPI(lifespan=lifespan, title="PDF-QA with FAISS+Gemini", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.post("/hackrx/run")
def run_submission(request_data: QueryRequest) -> Dict[str, Any]:
    try:
        results = logic.process_document_and_questions(
            pdf_url=request_data.documents,
            questions=request_data.questions
        )
        return results
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "API is running"}
