# logic.py
import os
import json
import re
import hashlib
import requests
import fitz
import numpy as np
from urllib.parse import urlparse
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document

from db import db_pool

import faiss

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set in env.")

# LLM + embeddings (Google/Gemini via LangChain adapter)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=GEMINI_API_KEY)

# ----------------- Helpers -----------------
def normalize_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def extract_file_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    return os.path.basename(parsed.path)

def normalize_questions(questions: list) -> list:
    cleaned = [re.sub(r'\s+', ' ', q.strip().lower()) for q in questions]
    return sorted(cleaned)

def get_documents_from_pdf_url(pdf_url, timeout=60):
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=timeout)
        response.raise_for_status()
        pdf_doc = fitz.open(stream=response.content, filetype="pdf")
        documents = [
            Document(page_content=page.get_text(), metadata={"source_page": i + 1})
            for i, page in enumerate(pdf_doc) if page.get_text().strip()
        ]
        pdf_doc.close()
        return documents
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def get_text_chunks(documents, chunk_size=1200, chunk_overlap=200):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return splitter.split_documents(documents)

def llm_parser_extract_query_topic(user_question):
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

# ----------------- DB helpers -----------------
def ensure_table_exists():
    if not db_pool:
        print("DB pool not initialized. Skipping table ensure.")
        return
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hackathon_cache (
                cache_key CHAR(32) PRIMARY KEY,
                pdf_url TEXT NOT NULL,
                file_name TEXT,
                answers JSONB NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        conn.commit()
        # add file_name column if missing (idempotent)
        cur.execute("""
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_name='hackathon_cache' AND column_name='file_name'
                ) THEN
                    ALTER TABLE hackathon_cache ADD COLUMN file_name TEXT;
                END IF;
            END$$;
        """)
        conn.commit()
        cur.close()
    except Exception as e:
        print(f"ensure_table_exists error: {e}")
    finally:
        if conn:
            db_pool.putconn(conn)

# ----------------- FAISS helpers -----------------
def build_faiss_index_from_embeddings(emb_list):
    """
    emb_list: list[list[float]] or numpy array shape (N, D)
    returns (index, normalized_vectors)
    """
    arr = np.array(emb_list).astype("float32")
    # cosine similarity via inner product after L2-normalize
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    arr_norm = arr / norms
    d = arr_norm.shape[1]
    index = faiss.IndexFlatIP(d)  # inner product for cosine
    index.add(arr_norm)
    return index, arr_norm

def search_index(index, query_vec, top_k=5):
    q = np.array(query_vec).astype("float32")
    q = q / (np.linalg.norm(q) + 1e-9)
    D, I = index.search(q.reshape(1, -1), top_k)
    return I[0], D[0]

# ----------------- Main pipeline -----------------
def process_document_and_questions(pdf_url, questions):
    """
    pdf_url: URL string
    questions: list[str]
    returns: {"answers": [...]}
    """
    # normalize
    pdf_url_normalized = normalize_pdf_url(pdf_url)
    file_name = extract_file_name_from_url(pdf_url)
    questions_normalized = normalize_questions(questions)
    question_string = "||".join(questions_normalized)
    cache_key = hashlib.md5((pdf_url_normalized + question_string).encode()).hexdigest()

    print(f"DEBUG: Normalized URL --> {pdf_url_normalized}")
    print(f"DEBUG: File name --> {file_name}")
    print(f"DEBUG: Normalized Questions --> {questions_normalized}")
    print(f"DEBUG: Cache key --> {cache_key}")

    # ensure table exists
    ensure_table_exists()

    # --- Check DB cache ---
    if db_pool:
        conn = None
        try:
            conn = db_pool.getconn()
            print(f"DEBUG: Connected to DB: {getattr(conn, 'dsn', 'unknown')}")
            cur = conn.cursor()
            cur.execute("SELECT answers, pdf_url, file_name FROM hackathon_cache WHERE cache_key = %s", (cache_key,))
            row = cur.fetchone()
            cur.close()
            if row:
                print(f"DATABASE CACHE HIT! Returning saved answer for key: {cache_key}")
                data = row[0]
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except Exception:
                        print("Warning: Could not parse JSON from DB; returning raw.")
                return data
        except Exception as e:
            print(f"Database cache check failed: {e}")
        finally:
            if conn:
                db_pool.putconn(conn)

    print(f"DATABASE CACHE MISS! Processing new request for key: {cache_key}")

    # Download PDF and parse
    documents = get_documents_from_pdf_url(pdf_url)
    if not documents:
        return {"answers": ["Failed to read PDF."] * len(questions)}

    text_chunks = get_text_chunks(documents)
    if not text_chunks:
        return {"answers": ["Failed to chunk text."] * len(questions)}

    chunk_texts = [chunk.page_content for chunk in text_chunks]

    # Get embeddings for chunks using Gemini embedding model
    try:
        print("Embedding document chunks...")
        chunk_embeddings = embeddings.embed_documents(chunk_texts)
        # chunk_embeddings: list of lists -> convert to numpy
    except Exception as e:
        print(f"Embedding error: {e}")
        return {"answers": ["Embedding error."] * len(questions)}

    # Build FAISS index
    try:
        index, arr_norm = build_faiss_index_from_embeddings(chunk_embeddings)
    except Exception as e:
        print(f"FAISS build error: {e}")
        return {"answers": ["FAISS error."] * len(questions)}

    final_simple_answers = []
    for question in questions:
        # Optionally extract short topic for better semantic search
        transformed_query = llm_parser_extract_query_topic(question)
        try:
            q_emb = embeddings.embed_query(transformed_query)
        except Exception as e:
            print(f"Query embedding error: {e}")
            q_emb = embeddings.embed_query(question)

        top_k = min(5, len(chunk_embeddings))
        idxs, scores = search_index(index, q_emb, top_k=top_k)

        retrieved_docs = [text_chunks[i] for i in idxs]

        # Build context JSON: page -> concatenated texts
        context_with_sources = {}
        for doc in retrieved_docs:
            # Try to find nearest doc's metadata page (we have chunk_texts mapping)
            # We don't have the original doc object here; map by equality
            # Find index in chunk_texts
            try:
                i = chunk_texts.index(doc)
                src_page = text_chunks[i].metadata.get("source_page", "Unknown")
            except Exception:
                src_page = "Unknown"
            context_with_sources.setdefault(str(src_page), []).append(doc)
        context_json_str = json.dumps(context_with_sources, indent=2)

        structured_answer = generate_structured_answer(context_json_str, question)
        final_simple_answers.append(structured_answer.get("answer", "Error."))

    final_response = {"answers": final_simple_answers}

    # Save to DB (answers JSONB) with autocommit and verify
    if db_pool:
        conn = None
        try:
            conn = db_pool.getconn()
            conn.autocommit = True
            cur = conn.cursor()
            print(f"DEBUG: Saving cache row for key: {cache_key}")
            cur.execute(
                """
                INSERT INTO hackathon_cache (cache_key, pdf_url, file_name, answers)
                VALUES (%s, %s, %s, %s::jsonb)
                ON CONFLICT (cache_key) DO UPDATE
                  SET answers = EXCLUDED.answers,
                      pdf_url = EXCLUDED.pdf_url,
                      file_name = EXCLUDED.file_name,
                      created_at = CURRENT_TIMESTAMP
                """,
                (cache_key, pdf_url_normalized, file_name, json.dumps(final_response))
            )
            # verify
            cur.execute("SELECT 1 FROM hackathon_cache WHERE cache_key = %s", (cache_key,))
            if cur.fetchone():
                print(f"DEBUG: Row successfully inserted/exists for key: {cache_key}")
            else:
                print(f"ERROR: Insert verification failed for key: {cache_key}")
            cur.close()
        except Exception as e:
            print(f"Database cache write failed: {e}")
        finally:
            if conn:
                db_pool.putconn(conn)

    return final_response
