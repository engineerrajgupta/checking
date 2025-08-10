from urllib.parse import urlparse
import re

def normalize_pdf_url(url: str) -> str:
    """Remove query params and fragments from PDF URL for consistent caching."""
    parsed = urlparse(url)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"

def normalize_questions(questions: list[str]) -> list[str]:
    """Trim whitespace, lowercase, and sort questions for consistent caching."""
    cleaned = [re.sub(r'\s+', ' ', q.strip().lower()) for q in questions]
    return sorted(cleaned)

def process_document_and_questions(pdf_url, questions):
    # --- Normalize input for consistent cache keys ---
    pdf_url_normalized = normalize_pdf_url(pdf_url)
    questions_normalized = normalize_questions(questions)
    question_string = "||".join(questions_normalized)
    cache_key = hashlib.md5((pdf_url_normalized + question_string).encode()).hexdigest()

    print(f"DEBUG: Normalized URL --> {pdf_url_normalized}")
    print(f"DEBUG: Normalized Questions --> {questions_normalized}")
    print(f"DEBUG: Cache key --> {cache_key}")

    # --- CACHE CHECK ---
    if db_pool:
        conn = None
        try:
            conn = db_pool.getconn()
            print(f"DEBUG: Connected to DB: {conn.dsn}")

            cur = conn.cursor()
            # Debug: See all cache keys in DB
            cur.execute("SELECT cache_key FROM hackathon_cache")
            all_keys = [row[0] for row in cur.fetchall()]
            print(f"DEBUG: Keys in DB: {all_keys}")

            # Try fetching our key
            cur.execute("SELECT answers FROM hackathon_cache WHERE cache_key = %s", (cache_key,))
            result = cur.fetchone()
            cur.close()

            if result:
                print(f"DATABASE CACHE HIT! Returning saved answer for key: {cache_key}")
                data = result[0]
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        print("Warning: Failed to parse JSON from cache.")
                return data
        except Exception as e:
            print(f"Database cache check failed: {e}")
        finally:
            if conn:
                db_pool.putconn(conn)

    print(f"DATABASE CACHE MISS! Processing new request for key: {cache_key}")

    # === Normal processing (unchanged) ===
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
            np.dot(query_embedding, chunk_emb) /
            (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb))
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

    # --- SAVE TO CACHE ---
    if db_pool:
        conn = None
        try:
            conn = db_pool.getconn()
            conn.autocommit = True  # Ensure immediate commit
            print(f"DEBUG: Writing to DB: {conn.dsn}")

            cur = conn.cursor()
            print(f"SAVING TO DATABASE CACHE for key: {cache_key}")
            cur.execute(
                """
                INSERT INTO hackathon_cache (cache_key, pdf_url, answers)
                VALUES (%s, %s, %s::jsonb)
                ON CONFLICT (cache_key) DO NOTHING
                """,
                (cache_key, pdf_url_normalized, json.dumps(final_response))
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
