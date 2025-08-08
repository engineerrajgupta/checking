# logic.py (FINAL - UNIFIED VERSION)

import os
import json
import requests
import io
import fitz  # To render PDF pages into images for Vision API
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
import numpy as np

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# --- Initialize LLM and Embeddings ---
# UPGRADED: Using the most powerful models to execute your proven prompts
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", google_api_key=api_key, temperature=0)
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-001", google_api_key=api_key)


# --- Core Logic Functions ---

# UPGRADED: Using Gemini Vision for intelligent, OCR-like text extraction
def get_documents_from_pdf_with_vision(pdf_url):
    """
    Downloads a PDF and uses Gemini Vision to perform intelligent OCR on each page.
    It returns a list of Document objects, one for each page, with metadata.
    """
    try:
        print(f"Downloading PDF from: {pdf_url}")
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_bytes = response.content
        pdf_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        documents = []
        print(f"Processing {len(pdf_doc)} pages with Gemini Vision...")
        for i, page in enumerate(pdf_doc):
            pix = page.get_pixmap(dpi=200)
            img_bytes = pix.tobytes("jpeg")
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')

            # Call the Vision API to extract text from the image
            response = llm.invoke([
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Extract all text from this document page. Preserve the layout and structure, especially for tables."},
                        {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_base64}"}
                    ]
                }
            ])
            page_text = response.content
            if page_text:
                documents.append(Document(page_content=page_text, metadata={"source_page": i + 1}))
        
        print("Vision processing complete.")
        return documents

    except Exception as e:
        print(f"An error occurred during Vision processing: {e}")
        return None

# KEPT: Using a better text splitter for the high-quality extracted text
def get_text_chunks(documents):
    """Splits Document objects into smaller chunks for processing."""
    text_splitter = RecursiveCharacterTextSplitter(
        separator="\n", chunk_size=1200, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_documents(documents)

# KEPT FROM YOUR BEST CODE: This function is part of the winning formula
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

# KEPT FROM YOUR BEST CODE: This is the high-performance prompt
def generate_structured_answer(context_with_sources, question):
    """
    Generates a structured JSON answer. This prompt structure forces the LLM
    to find evidence before answering, which has proven to increase accuracy.
    """
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
        print(f"An error occurred during LLM call: {e}")
        return {"answer": "Failed to get a response from the language model.", "source_quote": "N/A", "source_page_number": "N/A"}

def process_document_and_questions(pdf_url, questions):
    """Main processing pipeline combining Vision with your proven prompt logic."""
    documents = get_documents_from_pdf_with_vision(pdf_url)
    if not documents:
        return {"answers": ["Failed to process the document with the Vision API."] * len(questions)}

    text_chunks = get_text_chunks(documents)
    if not text_chunks:
        return {"answers": ["Failed to chunk the document text."] * len(questions)}

    chunk_texts = [chunk.page_content for chunk in text_chunks]
    chunk_embeddings = embeddings.embed_documents(chunk_texts)

    final_simple_answers = []

    for question in questions:
        # Use the query transformation from your best code
        transformed_query = llm_parser_extract_query_topic(question)
        query_embedding = embeddings.embed_query(transformed_query)

        similarities = [np.dot(query_embedding, chunk_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_emb)) for chunk_emb in chunk_embeddings]
        
        top_k = 8 # Use more context for the powerful Pro model
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        retrieved_docs = [text_chunks[i] for i in top_indices]
        
        # Format the context exactly as your winning prompt expects
        context_with_sources = {}
        for doc in retrieved_docs:
            page = doc.metadata.get("source_page", "Unknown")
            if page not in context_with_sources:
                context_with_sources[page] = []
            context_with_sources[page].append(doc.page_content)
        context_json_str = json.dumps(context_with_sources, indent=2)

        if retrieved_docs:
            structured_answer = generate_structured_answer(context_json_str, question)
            final_simple_answers.append(structured_answer.get("answer", "Error processing this question."))
        else:
            final_simple_answers.append("Could not find any relevant context for this question.")

    final_response = {"answers": final_simple_answers}
    print("\n--- FINAL API RESPONSE (as sent to judge) ---")
    print(json.dumps(final_response, indent=2))
    return final_response
