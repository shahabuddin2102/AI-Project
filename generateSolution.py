# ===== Helper Function For MCP Server Implementations =====
import os
import re
import requests
import logging
from typing import List, Dict, Any
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from fastapi import UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from settings import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, SOLUTIONS_LOG_FILE, SOLUTION_PROMPT_PATH

from logger_setup import setup_logger

logger = setup_logger("GroqServer", SOLUTIONS_LOG_FILE)


if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is required")

if not GROQ_MODEL:
    raise ValueError("GROQ_MODEL environment variable is required")

# ===== Models =====
class ProcessingResult(BaseModel):
    query: str
    solution: str

def load_prompt_template() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # prompt_path = os.path.join(base_dir, "prompts", "solution_prompt.txt")
    prompt_path = SOLUTION_PROMPT_PATH
    
    if not os.path.exists(prompt_path):
        logger.error(f"Prompt file not found: {prompt_path}")
        raise RuntimeError("Prompt file is missing.")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load prompt file: {e}")
        raise RuntimeError("Prompt file is unreadable.")

def process_text_content(text_content: str, chunk_size: int = 1000, chunk_overlap: int = 150) -> List[Document]:
    logger.info("Splitting text content into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents([Document(page_content=text_content)])

def call_groq_api(context: str) -> Dict[str, str]:
    logger.info("Calling Groq API with chunk content...")
    system_prompt = load_prompt_template()

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}"}
        ],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        response = requests.post(
            GROQ_BASE_URL,
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()

        logger.info("Groq API response received successfully")
        result = response.json()["choices"][0]["message"]["content"]

        query_match = re.search(r"Query:\s*(.+?)(?=\nSolution:|\n\n|$)", result, re.DOTALL)
        solution_match = re.search(r"Solution:\s*(.+)", result, re.DOTALL)

        return {
            "query": query_match.group(1).strip() if query_match else "No query generated",
            "solution": solution_match.group(1).strip() if solution_match else "No solution generated"
        }

    except Exception as e:
        logger.error(f"Groq API error: {str(e)}", exc_info=True)
        return {
            "query": "Error generating query",
            "solution": f"Error: {str(e)}"
        }

def process_chunks(documents: List[Document], chunk_limit: int) -> List[ProcessingResult]:
    logger.info(f"Processing up to {chunk_limit} chunks...")
    results = []
    for i, doc in enumerate(documents[:chunk_limit]):
        try:
            result = call_groq_api(doc.page_content.strip())
            results.append(ProcessingResult(
                query=result["query"],
                solution=result["solution"],
            ))
        except Exception as e:
            logger.error(f"Error processing chunk {i + 1}: {str(e)}")
            continue
    logger.info(f"Finished processing {len(results)} chunks.")
    return results

