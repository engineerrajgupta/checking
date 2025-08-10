# logic.py
import os
import json
import requests
import fitz
import hashlib
import numpy as np
import re
from urllib.parse import urlparse
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document

from db import db_pool

# --- Load Environment & Models ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# NOTE: ensure the langchain-google-genai package works in your environment.
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", google_api_key=api_key)

# --- Helpers ---
def normalize_pdf_url(url: str) -> str:
    """Return scheme://netloc/path (strip query & fragment)."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def extract_file_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    return os.path.basename(parsed.path)

def normalize_questions(questions: list) -> list:
    """Trim whitespace, lower, compress multiple spaces, sort."""
    cleaned = [re.sub(r'\s+', ' ', q.strip().lower()) for q in questions]
    return sorted(cleaned)

# --- DB table setup (create/alter as needed) ---
def setup_database():
    if not db_pool:
        print("Database pool not available. Skipping table setup.")
        return
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        # Create table if not exists
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
        # Ensure columns exist (idempotent)
        # (This ensures older schema without file_name will be upgraded)
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
        print("Database table 'hackathon_cache' is ready (created/altered if needed).")
    except Exception as e:
        print(f"Database setup failed: {e}")
    finally:
        if conn:
            db_pool.putconn(conn)

# --- PDF & LLM functions ---
def get_documents_from_pdf_url(pdf_url):
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
        pdf_doc = fitz.open(stream=response.content, filetype="pdf")
        documents = [
            Document(page_content=page.get_text(), metadata={"source_page": i + 1})
            for i, page in enumerate(pdf_doc) if page.get_text()
        ]
        pdf_doc.close()
        return documents
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return None

def get_text_chunks(documents):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1200, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_documents(documents)

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

# --- Main processing pipeline ---
def process_document_and_questions(pdf_url, questions):
    # Normalize input
    pdf_url_normalized = normalize_pdf_url(pdf_url)
    file_name = extract_file_name_from_url(pdf_url)
    questions_normalized = normalize_questions(questions)
    question_string = "||".join(questions_normalized)
    cache_key = hashlib.md5((pdf_url_normalized + question_string).encode()).hexdigest()

    print(f"DEBUG: Normalized URL --> {pdf_url_normalized}")
    print(f"DEBUG: File name --> {file_name}")
    print(f"DEBUG: Normalized Questions --> {questions_normalized}")
    print(f"DEBUG: Cache key --> {cache_key}")

    # --- Cache check ---
    if db_pool:
        conn = None
        try:
            conn = db_pool.getconn()
            print(f"DEBUG: Connected to DB: {getattr(conn, 'dsn', 'unknown dsn')}")
            cur = conn.cursor()

            # Debug: list keys (for debugging only - remove if huge)
            try:
                cur.execute("SELECT cache_key, pdf_url, file_name, created_at FROM hackathon_cache ORDER BY created_at DESC LIMIT 20;")
                rows = cur.fetchall()
                print("DEBUG: Recent rows (cache_key, pdf_url, file_name, created_at):")
                for r in rows:
                    print(r)
            except Exception as e:
                print(f"DEBUG: Could not fetch recent rows: {e}")

            cur.execute("SELECT answers, pdf_url, file_name FROM hackathon_cache WHERE cache_key = %s", (cache_key,))
            result = cur.fetchone()
            cur.close()

            if result:
                print(f"DATABASE CACHE HIT! Returning saved answer for key: {cache_key}")
                data = result[0]
                # If answers stored as string, parse it
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except Exception:
                        print("Warning: Failed to parse JSON from cache; returning raw string.")
                # Optionally also show where it came from:
                cached_pdf_url = result[1]
                cached_file_name = result[2]
                print(f"DEBUG: Hit metadata -> pdf_url: {cached_pdf_url}, file_name: {cached_file_name}")
                return data
        except Exception as e:
            print(f"Database cache check failed: {e}")
        finally:
            if conn:
                db_pool.putconn(conn)

    print(f"DATABASE CACHE MISS! Processing new request for key: {cache_key}")

    # --- Process PDF & query embeddings + LLM ---
    documents = get_documents_from_pdf_url(pdf_url)
    if not documents:
        return {"answers": ["Failed to read PDF."] * len(questions)}

    text_chunks = get_text_chunks(documents)
    if not text_chunks:
        return {"answers": ["Failed to chunk text."] * len(questions)}

    chunk_texts = [chunk.page_content for chunk in text_chunks]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)

    final_simple_answers = []
    for question in questions:
        transformed_query = llm_parser_extract_query_topic(question)
        query_embedding = embeddings.embed_query(transformed_query)

        similarities = [
            np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
            for chunk_emb in chunk_embeddings
        ]

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

    # --- Save to DB with file metadata ---
    if db_pool:
        conn = None
        try:
            conn = db_pool.getconn()
            # Force immediate commit for this connection
            conn.autocommit = True
            print(f"DEBUG: Writing to DB: {getattr(conn, 'dsn', 'unknown dsn')}")
            cur = conn.cursor()
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

            # Verify insert
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
