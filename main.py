# main.py (FINAL - WITH LIFESPAN FIX)

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# UPDATED: Import both the main function and the setup function from logic.py
from logic import process_document_and_questions, setup_database

# --- Lifespan Event ---
# This special function runs code ONCE when the application starts up.
# This is the professional way to initialize resources like a database connection.
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Application startup...")
    setup_database()  # Run the database setup at the correct time
    yield
    print("Application shutdown...")

# --- FastAPI App Initialization ---
# We tell FastAPI to use our new lifespan function to manage startup.
app = FastAPI(
    title="Intelligent Query-Retrieval API",
    description="An API that uses a persistent database cache for ultimate performance.",
    version="12.0.0-lifespan-fix",
    lifespan=lifespan
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Body Validation ---
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# --- API Endpoint (Unchanged) ---
@app.post("/hackrx/run")
def run_submission(request_data: QueryRequest) -> Dict[str, Any]:
    """
    This endpoint accepts a PDF document URL and a list of questions.
    It processes them using the logic in logic.py and returns a JSON response.
    """
    print("Received request for /hackrx/run")
    try:
        results = process_document_and_questions(
            pdf_url=request_data.documents, 
            questions=request_data.questions
        )
        
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
            
        print("Successfully processed request. Returning simple results.")
        return results

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Health Check Endpoint ---
@app.get("/")
def read_root():
    return {"status": "API is running. Send POST requests to /hackrx/run"}

# To run this server locally, use the command: uvicorn main:app --reload
