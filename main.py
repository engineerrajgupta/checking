# main.py (Final Version for Database Logic)

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

# Import the correct, non-async function name from our final logic.py
from logic import process_document_and_questions

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Intelligent Query-Retrieval API",
    description="An API that uses a persistent database cache for ultimate performance.",
    version="11.0.0-db-cache"
)

# --- CORS (Cross-Origin Resource Sharing) Middleware ---
# This allows the API to be called from any web front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Pydantic Models for Request Body Validation ---
# This ensures the incoming data has the correct structure
class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


# --- API Endpoint ---
@app.post("/hackrx/run")
def run_submission(request_data: QueryRequest) -> Dict[str, Any]:
    """
    This endpoint accepts a PDF document URL and a list of questions.
    It processes them using the logic in logic.py and returns a JSON response.
    """
    print("Received request for /hackrx/run")
    try:
        # Call the standard (non-async) function from logic.py
        results = process_document_and_questions(
            pdf_url=request_data.documents, 
            questions=request_data.questions
        )
        
        # Basic error handling
        if "error" in results:
            raise HTTPException(status_code=400, detail=results["error"])
            
        print("Successfully processed request. Returning simple results.")
        return results

    except Exception as e:
        # Catch-all for any unexpected errors during processing
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- Health Check Endpoint ---
# A simple endpoint to confirm the API is running
@app.get("/")
def read_root():
    return {"status": "API is running. Send POST requests to /hackrx/run"}

# To run this server locally, use the command: uvicorn main:app --reload
