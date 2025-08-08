# logic.py (FINAL - HYBRID PARSER + DATABASE CACHE)

import os
import json
import requests
import io
import fitz
import hashlib
import base64 # For Vision API
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np
from db import db_pool # Using the connection pool

# --- Load Environment Variables & Models ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0.2)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)

# --- Database Setup (from db.py, called by main.py) ---
def setup_database():
    """Uses a connection from the pool to create the cache table if it doesn't exist."""
    if not db_pool:
        print("Database pool not available. Skipping table setup.")
        return
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hackathon_cache (
                cache_key CHAR(32) PRIMARY KEY,
                pdf_url TEXT NOT NULL,
                answers JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        cur.close()
        print("Database table 'hackathon_cache' is ready.")
    except Exception as e:
        print(f"Database setup failed: {e}")
    finally:
        if conn:
            db_pool.putconn(conn)

# --- THE HYBRID PARSER (Prevents Crashes) ---
def get_documents_from_pdf_url(pdf_url):
    """
    Downloads and intelligently parses a PDF. It first tries a fast text extraction.
    If it detects a scanned/image-based PDF, it falls back to the powerful Vision API.
    """
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_bytes = response.content
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        documents = []
        total_text_length = 0
        for i, page in enumerate(pdf_doc):
            text = page.get_text("text", sort=True)
            total_text_length += len(text)
            if text:
                documents.append(Document(page_content=text, metadata={"source_page": i + 1}))
        
        # Heuristic: If the average characters per page is less than 100, it's likely a scan.
        if len(pdf_doc) > 0 and total_text_length / len(pdf_doc) < 100:
            print(f"LOW TEXT DETECTED ({total_text_length} chars / {len(pdf_doc)} pages). Falling back to Vision API.")
            documents = [] # Discard the garbage text
            for i, page in enumerate(pdf_doc):
                pix = page.get_pixmap(dpi=200)
                img_bytes = pix.tobytes("jpeg")
                img_base64 = base64.b64encode(img_bytes).decode('utf-8')
                
                # Use the 'llm' object which is gemini-1.5-flash, it can handle vision too
                response = llm.invoke([
                    {"role": "user", "content": [
                        {"type": "text", "text": "Extract all text from this document page. Preserve layout."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                    ]}
                ])
                page_text = response.content
                if page_text:
                    documents.append(Document(page_content=page_text, metadata={"source_page": i + 1}))
            print("Vision processing complete.")
        else:
            print("Standard text extraction successful.")
        
        pdf_doc.close()
        return documents
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

# --- Your Proven Logic (Unchanged) ---
def get_text_chunks(documents):
    """Splits Document objects into smaller chunks for processing."""
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1200, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_documents(documents)

def llm_parser_extract_query_topic(user_question):
    """Uses the LLM to parse the user's question and extract the core topic."""
    prompt = f"""
    You are an expert at identifying the core subject of a question.
    Analyze the following user question and extract its main topic for semantic search.
    User question: "{user_question}"
    Return a JSON object with a single key "query_topic".
    Respond ONLY with the JSON object.
    """
    try:
        response = llm.invoke(prompt)
        json_string = response.content.strip().replace("```json", "").replace("```", "")
        parsed_json = json.loads(json_string)
        return parsed_json.get("query_topic", user_question)
    except Exception:
        return user_question

def generate_structured_answer(context_with_sources, question):
    """Generates a structured JSON answer using your proven, high-performance prompt."""
    prompt = f"""
    You are a highly intelligent logic engine for analyzing legal and insurance documents.
    Your task is to answer the user's question based STRICTLY on the provided context.
    The context is a JSON object where keys are page numbers and values are the text from those pages.
    You must generate a structured JSON response.

    **Provided Context from Document:**
    ---
    {context_with_sources}
    ---

    **User's Question:**
    ---
    {question}
    ---

    **Your Task:**
    1. Find the single most relevant page and quote that answers the question.
    2. Generate a JSON object with the following schema:
    {{
      "question": "{question}",
      "answer": "A concise, direct answer to the question.",
      "source_quote": "The single, most relevant sentence from the context that directly supports your answer.",
      "source_page_number": "The page number (as an integer) where the source_quote was found."
    }}

    If the information is not in the context, respond with this JSON structure:
    {{
      "question": "{question}",
      "answer": "Information not found in the provided document context.",
      "source_quote": "N/A",
      "source_page_number": "N/A"
    }}
    """
    try:
        response = llm.invoke(prompt)
        json_string = response.content.strip().replace("```json", "").replace("```", "")
        return json.loads(json_string)
    except Exception as e:
        print(f"LLM call error: {e}")
        return {"answer": "LLM Error", "source_quote": "N/A", "source_page_number": "N/A"}

# --- Main Processing Pipeline with the Connection Pool Bug Fix ---
def process_document_and_questions(pdf_url, questions):
    """Main processing pipeline with the persistent and efficient database caching strategy."""
    
    question_string = "".join(sorted(questions))
    cache_key = hashlib.md5((pdf_url + question_string).encode()).hexdigest()
    
    print(f"DEBUG: App is looking for this exact key --> {cache_key}")

    if db_pool:
        conn = None
        result_to_return = None
        try:
            conn = db_pool.getconn()
            cur = conn.cursor()
            cur.execute("SELECT answers FROM hackathon_cache WHERE cache_key = %s", (cache_key,))
            result = cur.fetchone()
            cur.close()
            if result:
                print(f"DATABASE CACHE HIT! Returning saved answer for key: {cache_key}")
                result_to_return = json.loads(result[0])
        except Exception as e:
            print(f"Database cache check failed: {e}")
        finally:
            if conn:
                db_pool.putconn(conn)
        
        if result_to_return:
            return result_to_return
    
    print(f"DATABASE CACHE MISS! Processing new request for key: {cache_key}")
    
    documents = get_documents_from_pdf_url(pdf_url)
    if not documents: return {"answers": ["Failed to read PDF."] * len(questions)}

    text_chunks = get_text_chunks(documents)
    if not text_chunks: return {"answers": ["Failed to chunk text."] * len(questions)}

    chunk_texts = [chunk.page_content for chunk in text_chunks]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)

    final_simple_answers = []
    for question in questions:
        transformed_query = llm_parser_extract_query_topic(question)
        query_embedding = embeddings.embed_query(transformed_query)
        similarities = [np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)) for chunk_emb in chunk_embeddings]
        top_k = 5
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        retrieved_docs = [text_chunks[i] for i in top_indices]
        
        context_with_sources = {}
        for doc in retrieved_docs:
            page = doc.metadata.get("source_page", "Unknown")
            if page not in context_with_sources:
                context_with_sources[page] = []
            context_with_sources[page].append(doc.page_content)
        context_json_str = json.dumps(context_with_sources, indent=2)

        if retrieved_docs:
            structured_answer = generate_structured_answer(context_json_str, question)
            final_simple_answers.append(structured_answer.get("answer", "Error."))
        else:
            final_simple_answers.append("No relevant context found.")

    final_response = {"answers": final_simple_answers}
    
    if db_pool:
        conn = None
        try:
            conn = db_pool.getconn()
            cur = conn.cursor()
            print(f"SAVING TO DATABASE CACHE for key: {cache_key}")
            cur.execute(
                "INSERT INTO hackathon_cache (cache_key, pdf_url, answers) VALUES (%s, %s, %s) ON CONFLICT (cache_key) DO NOTHING",
                (cache_key, pdf_url, json.dumps(final_response))
            )
            conn.commit()
            cur.close()
        except Exception as e:
            print(f"Database cache write failed: {e}")
        finally:
            if conn:
                db_pool.putconn(conn)
            
    return final_response
