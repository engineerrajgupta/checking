# logic.py
import os
import io
import json
import re
import hashlib
import pickle
import requests
import fitz
import numpy as np
import faiss
from urllib.parse import urlparse
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain.docstore.document import Document

from db import db_pool  # expects your db.py to provide db_pool SimpleConnectionPool

load_dotenv()

# --------- Configuration ---------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set in environment.")

LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "gemini-embedding-001")

# Where to persist per-PDF FAISS indexes & metadata
FAISS_DIR = os.getenv("FAISS_DIR", "faiss_indexes")
PDF_CACHE_DIR = os.getenv("PDF_CACHE_DIR", "pdf_cache")
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(PDF_CACHE_DIR, exist_ok=True)

# instantiate models
llm = ChatGoogleGenerativeAI(model=LLM_MODEL, google_api_key=GEMINI_API_KEY, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, google_api_key=GEMINI_API_KEY)

# --------- Helpers ---------
def normalize_pdf_url(url: str) -> str:
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def extract_file_name_from_url(url: str) -> str:
    parsed = urlparse(url)
    return os.path.basename(parsed.path) or "document.pdf"

def normalize_questions(questions: List[str]) -> List[str]:
    cleaned = [re.sub(r'\s+', ' ', q.strip().lower()) for q in questions]
    return sorted(cleaned)

def md5_hex(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

# --------- PDF download & parsing ---------
def download_pdf_if_needed(pdf_url: str) -> str:
    """
    Download the pdf to PDF_CACHE_DIR and return local path.
    Uses md5(filename) as storage filename to handle query strings.
    """
    normalized = normalize_pdf_url(pdf_url)
    file_hash = md5_hex(normalized)
    filename = f"{file_hash}.pdf"
    local_path = os.path.join(PDF_CACHE_DIR, filename)
    if os.path.exists(local_path):
        print(f"Using cached PDF file at {local_path}")
        return local_path

    try:
        print(f"Downloading PDF from: {pdf_url}")
        resp = requests.get(pdf_url, timeout=60)
        resp.raise_for_status()
        with open(local_path, "wb") as f:
            f.write(resp.content)
        print(f"Saved PDF to {local_path}")
        return local_path
    except Exception as e:
        print(f"Error downloading PDF: {e}")
        if os.path.exists(local_path):
            os.remove(local_path)
        return None

def extract_documents_from_pdf(local_pdf_path: str) -> List[Document]:
    try:
        pdf_doc = fitz.open(local_pdf_path)
        documents = [
            Document(page_content=page.get_text(), metadata={"source_page": i + 1})
            for i, page in enumerate(pdf_doc) if page.get_text().strip()
        ]
        pdf_doc.close()
        return documents
    except Exception as e:
        print(f"Error parsing PDF {local_pdf_path}: {e}")
        return []

# --------- Text chunking ---------
def get_text_chunks_from_documents(documents: List[Document], chunk_size=1200, chunk_overlap=200) -> List[Document]:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    return text_splitter.split_documents(documents)

# --------- FAISS index persistence helpers ---------
def faiss_paths_for_pdf_hash(pdf_hash: str):
    index_path = os.path.join(FAISS_DIR, f"{pdf_hash}.index")
    meta_path = os.path.join(FAISS_DIR, f"{pdf_hash}.meta.pkl")
    return index_path, meta_path

def save_faiss_index(index: faiss.Index, metadata: List[dict], pdf_hash: str):
    index_path, meta_path = faiss_paths_for_pdf_hash(pdf_hash)
    faiss.write_index(index, index_path)
    with open(meta_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved FAISS index to {index_path} and metadata to {meta_path}")

def load_faiss_index(pdf_hash: str):
    index_path, meta_path = faiss_paths_for_pdf_hash(pdf_hash)
    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        return None, None
    try:
        index = faiss.read_index(index_path)
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        print(f"Loaded FAISS index from {index_path}")
        return index, metadata
    except Exception as e:
        print(f"Failed to load FAISS index: {e}")
        return None, None

# --------- Build FAISS index (cosine via inner-product of L2-normalized vectors) ---------
def build_faiss_index(embeddings_list: List[List[float]]):
    arr = np.array(embeddings_list).astype("float32")
    # normalize each vector (L2)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1e-9
    arr_norm = arr / norms
    d = arr_norm.shape[1]
    index = faiss.IndexFlatIP(d)  # inner-product -> works as cosine after normalization
    index.add(arr_norm)
    return index, arr_norm

# --------- FAISS search helper ---------
def search_faiss(index: faiss.Index, query_vector: List[float], top_k: int = 5):
    q = np.array(query_vector).astype("float32")
    q = q / (np.linalg.norm(q) + 1e-9)
    D, I = index.search(q.reshape(1, -1), top_k)
    return I[0].tolist(), D[0].tolist()

# --------- LLM prompts / structured answer ----------
def llm_parser_extract_query_topic(user_question: str) -> str:
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
        parsed = json.loads(json_string)
        return parsed.get("query_topic", user_question)
    except Exception:
        return user_question

def generate_structured_answer(context_with_sources: str, question: str) -> dict:
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
        return {"question": question, "answer": "LLM Error", "source_quote": "N/A", "source_page_number": "N/A"}

# --------- DB caching helpers (Postgres JSONB) ----------
def ensure_cache_table_exists():
    if not db_pool:
        print("DB pool not initialized, skipping table ensure.")
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
        # Add file_name column if missing (idempotent)
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
        print(f"Error ensuring cache table: {e}")
    finally:
        if conn:
            db_pool.putconn(conn)

def fetch_from_cache(cache_key: str):
    if not db_pool:
        return None
    conn = None
    try:
        conn = db_pool.getconn()
        cur = conn.cursor()
        print(f"DEBUG: Running SELECT for cache_key={cache_key}")
        cur.execute("SELECT answers FROM hackathon_cache WHERE cache_key = %s", (cache_key,))
        row = cur.fetchone()
        cur.close()
        if row:
            data = row[0]
            # If driver returns string for JSONB, parse it
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except Exception:
                    print("Warning: Could not parse JSONB string from DB.")
            return data
        return None
    except Exception as e:
        print(f"DB fetch error: {e}")
        return None
    finally:
        if conn:
            db_pool.putconn(conn)

def save_to_cache(cache_key: str, pdf_url: str, file_name: str, final_response: dict):
    if not db_pool:
        print("DB pool not available; not saving cache.")
        return
    conn = None
    try:
        conn = db_pool.getconn()
        conn.autocommit = True
        cur = conn.cursor()
        print(f"DEBUG: Inserting cache_key={cache_key} into DB")
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
            (cache_key, pdf_url, file_name, json.dumps(final_response))
        )
        # verify insert
        cur.execute("SELECT 1 FROM hackathon_cache WHERE cache_key = %s", (cache_key,))
        ok = cur.fetchone()
        if ok:
            print(f"DEBUG: Cache saved / exists for key={cache_key}")
        else:
            print(f"ERROR: Cache verification failed for key={cache_key}")
        cur.close()
    except Exception as e:
        print(f"DB save error: {e}")
    finally:
        if conn:
            db_pool.putconn(conn)

# --------- Main processing pipeline ----------
def process_document_and_questions(pdf_url: str, questions: List[str]) -> Dict[str, Any]:
    """
    Main entrypoint.
    pdf_url: URL to PDF
    questions: list of user questions
    returns: {"answers": [ ... ]} where each answer is a concise string
    """
    # normalize inputs
    pdf_url_normalized = normalize_pdf_url(pdf_url)
    file_name = extract_file_name_from_url(pdf_url)
    questions_normalized = normalize_questions(questions)
    question_string = "||".join(questions_normalized)
    cache_key = md5_hex(pdf_url_normalized + question_string)

    print(f"DEBUG: Normalized URL --> {pdf_url_normalized}")
    print(f"DEBUG: File name --> {file_name}")
    print(f"DEBUG: Normalized Questions --> {questions_normalized}")
    print(f"DEBUG: Cache key --> {cache_key}")

    # ensure DB table exists
    ensure_cache_table_exists()

    # 1) Try cache
    cached = fetch_from_cache(cache_key)
    if cached:
        print(f"DATABASE CACHE HIT! Returning saved answer for key: {cache_key}")
        return cached

    print(f"DATABASE CACHE MISS! Processing new request for key: {cache_key}")

    # 2) Ensure PDF is downloaded locally
    local_pdf = download_pdf_if_needed(pdf_url)
    if not local_pdf:
        return {"answers": ["Failed to download PDF."] * len(questions)}

    # 3) Try loading existing FAISS index for this PDF
    pdf_hash = md5_hex(normalize_pdf_url(pdf_url))
    index, metadata = load_faiss_index(pdf_hash)

    chunk_texts = []
    chunk_metadata = []

    if index is None:
        # Need to parse PDF, chunk text, embed, and build index
        documents = extract_documents_from_pdf(local_pdf)
        if not documents:
            return {"answers": ["Failed to parse PDF."] * len(questions)}

        text_chunks = get_text_chunks_from_documents(documents)
        if not text_chunks:
            return {"answers": ["Failed to chunk text."] * len(questions)}

        # Keep chunk_texts and metadata lists aligned with embeddings
        chunk_texts = [chunk.page_content for chunk in text_chunks]
        chunk_metadata = [ {"source_page": chunk.metadata.get("source_page", "Unknown")} for chunk in text_chunks ]

        # compute embeddings via Gemini embeddings model
        try:
            print("Computing embeddings for document chunks...")
            chunk_embeddings = embeddings.embed_documents(chunk_texts)
            # build faiss index
            index, _ = build_faiss_index(chunk_embeddings)
            metadata = chunk_metadata
            save_faiss_index(index, metadata, pdf_hash)
        except Exception as e:
            print(f"Embedding / FAISS build error: {e}")
            return {"answers": ["Embedding/FAISS build failed."] * len(questions)}
    else:
        # Load chunk_texts and metadata from disk metadata
        chunk_texts = []
        # metadata is already loaded and contains source_page per chunk
        # we need the actual text chunks: we stored only metadata above, so we must load chunk texts too
        # To keep it simple, we persist chunk_texts inside metadata as well during save.
        # If metadata contains 'text' per chunk, use it; else we cannot reconstruct chunk texts
        if metadata and isinstance(metadata, list) and "text" in metadata[0]:
            chunk_texts = [m.get("text", "") for m in metadata]
        else:
            # Not having chunk text in metadata — fallback: re-parse & re-embed (safer)
            print("Metadata does not include chunk text; reparsing PDF to rebuild embeddings.")
            documents = extract_documents_from_pdf(local_pdf)
            text_chunks = get_text_chunks_from_documents(documents)
            chunk_texts = [chunk.page_content for chunk in text_chunks]
            chunk_metadata = [ {"source_page": chunk.metadata.get("source_page", "Unknown"), "text": chunk.page_content} for chunk in text_chunks ]
            try:
                chunk_embeddings = embeddings.embed_documents(chunk_texts)
                index, _ = build_faiss_index(chunk_embeddings)
                save_faiss_index(index, chunk_metadata, pdf_hash)
                metadata = chunk_metadata
            except Exception as e:
                print(f"Rebuild embeddings after missing chunk texts failed: {e}")
                return {"answers": ["Embedding rebuild failed."] * len(questions)}

    # If metadata lacks chunk text, ensure metadata now contains text fields for later context assembly
    if metadata and isinstance(metadata, list) and "text" not in metadata[0]:
        # attach chunk text to metadata based on currently available chunk_texts
        if len(metadata) == len(chunk_texts):
            for i, m in enumerate(metadata):
                m["text"] = chunk_texts[i]

    # 4) For each question: embed query, search FAISS, collect top chunks, ask LLM
    final_simple_answers = []
    for question in questions:
        # try to extract query topic for better retrieval
        transformed_query = llm_parser_extract_query_topic(question)
        try:
            q_emb = embeddings.embed_query(transformed_query)
        except Exception as e:
            print(f"Query embedding failed for transformed query; embedding original question. Error: {e}")
            q_emb = embeddings.embed_query(question)

        top_k = min(5, len(metadata))
        try:
            idxs, scores = search_faiss(index, q_emb, top_k=top_k)
        except Exception as e:
            print(f"FAISS search error: {e}")
            idxs, scores = [], []

        retrieved_context = {}
        for i in idxs:
            if i < 0 or i >= len(metadata):
                continue
            meta = metadata[i]
            page = meta.get("source_page", "Unknown")
            text = meta.get("text") or (chunk_texts[i] if i < len(chunk_texts) else "")
            if page not in retrieved_context:
                retrieved_context[page] = []
            retrieved_context[page].append(text)

        context_json_str = json.dumps(retrieved_context, indent=2)
        structured = generate_structured_answer(context_json_str, question)
        final_simple_answers.append(structured.get("answer", "Information not found."))

    final_response = {"answers": final_simple_answers}

    # 5) Save to DB cache
    try:
        save_to_cache(cache_key, pdf_url_normalized, file_name, final_response)
    except Exception as e:
        print(f"Warning: saving to DB failed: {e}")

    return final_response
