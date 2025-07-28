# === Helper Function For MCP Server Implementation ===
import os
import json
import logging
import re
import httpx
import asyncio
from typing import Any, Dict
from typing import Dict, List, Union
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from settings import GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, QA_LOG_FILE, QA_PROMPT_PATH
from logger_setup import setup_logger

logger = setup_logger("MCP-QA-Generator", QA_LOG_FILE)

if not GROQ_API_KEY and GROQ_MODEL:
    logger.error("GROQ_API_KEY & GROQ_MODEL is not set in environment variables")
    raise RuntimeError("Missing GROQ_API_KEY environment variable")


def load_prompt_template() -> str:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    # prompt_path = os.path.join(base_dir, "prompts", "qa_prompt.txt")
    prompt_path = QA_PROMPT_PATH
    
    if not os.path.exists(prompt_path):
        logger.error(f"Prompt file not found: {prompt_path}")
        raise RuntimeError("Prompt file is missing.")
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load prompt file: {e}")
        raise RuntimeError("Prompt file is unreadable.")


def build_prompt(text_chunk: str) -> str:
    try:
        prompt_template = load_prompt_template()

        if "{text}" not in prompt_template:
            raise ValueError("Prompt template is missing the '{text}' placeholder.")

        return prompt_template.replace("{text}", text_chunk)

    except Exception as e:
        logger.error(f"Error building prompt: {e}")
        raise

# === Text processing utilities ===
def chunk_text(text: str, max_chars=5000, overlap=500) -> List[str]:
    paragraphs = text.split("\n\n")
    chunks = []
    current = ""
    for para in paragraphs:
        if len(current) + len(para) <= max_chars:
            current += para + "\n\n"
        else:
            chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else ""
            if "\n" in overlap_text:
                overlap_text = overlap_text[overlap_text.rfind("\n")+1:]
            current = overlap_text + para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    logger.info(f"Chunked text into {len(chunks)} chunks")
    return chunks

async def generate_qa_from_text(prompt: str, retries=3, delay=1) -> str:
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": GROQ_MODEL,
        "temperature": 0.3,
        "max_tokens": 6000,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that extracts detailed, factual, and well-structured "
                    "question–answer pairs from input documents. You respond only with a valid JSON array, "
                    "each object containing 'question' and 'answer' keys. Never include text outside the array."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    }

    timeout_seconds = 60

    async with httpx.AsyncClient(timeout=timeout_seconds) as client:
        for attempt in range(retries):
            try:
                logger.debug(f"Groq API call attempt {attempt + 1}")
                response = await client.post(
                    GROQ_BASE_URL,
                    headers=headers,
                    json=data
                )

                if response.status_code == 200:
                    result = response.json()
                    return result['choices'][0]['message']['content'].strip()
                else:
                    logger.error(f"Groq API error: {response.status_code} - {response.text}")
            except httpx.ReadTimeout:
                logger.error(f"Groq API read timeout on attempt {attempt + 1}")
            except Exception as e:
                logger.error(f"Exception during Groq API call: {e}")

            await asyncio.sleep(delay * (2 ** attempt))

    raise RuntimeError("Groq LLM call failed after retries")


def extract_json_from_response(text: str) -> Union[List[Dict[str, str]], Dict]:
    try:
        match = re.search(r'\[\s*{.*?"question"\s*:\s*".+?".*?"answer"\s*:\s*".+?".*?}\s*]', text, re.DOTALL)
        return json.loads(match.group()) if match else {}
    except Exception as e:
        logger.error(f"JSON parsing error: {e}")
        return {}

def parse_and_merge_qa_responses(responses: List[str]) -> List[Dict[str, str]]:
    qa_list = []
    for response in responses:
        extracted = extract_json_from_response(response)
        if isinstance(extracted, list):
            for item in extracted:
                if isinstance(item, dict) and "question" in item and "answer" in item:
                    qa_list.append({
                        "question": item["question"].strip(),
                        "answer": item["answer"].strip()
                    })
    logger.info(f"Total Q&A pairs parsed: {len(qa_list)}")
    return qa_list


