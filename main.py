# main.py
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from logic import process_document_and_questions  # now it exists!

app = FastAPI(title="HackRx PDF Q&A (Gemini)")

class QueryRequest(BaseModel):
    documents: str
    questions: List[str]

@app.post("/hackrx/run")
async def run_submission(request_data: QueryRequest) -> Dict[str, Any]:
    try:
        result = process_document_and_questions(
            pdf_url=request_data.documents,
            questions=request_data.questions
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "API is running."}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
