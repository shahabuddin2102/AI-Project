import os
from dotenv import load_dotenv

load_dotenv()

# ===== General Settings =====
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===== GROQ API KEY & BASE URL =====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1/chat/completions")

# ===== Prompt Template Paths =====
QA_PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "qa_prompt.txt")
SOLUTION_PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "solution_prompt.txt")

# ===== Log File Paths =====
QA_LOG_FILE = os.path.join(BASE_DIR, "CZKB-LOGS", "generateqa_logs.log")
SOLUTIONS_LOG_FILE = os.path.join(BASE_DIR, "CZKB-LOGS", "generatesolution_logs.log")

# ===== QA Pagination Defaults =====
QA_DEFAULT_LIMIT = 10
QA_MIN_LIMIT = 1
QA_MAX_LIMIT = 100 
QA_DEFAULT_START = 0         

# ===== Solution Pagination Defaults =====
SOLUTION_DEFAULT_LIMIT = 10
SOLUTION_MIN_LIMIT = 1
SOLUTION_MAX_LIMIT = 100
SOLUTION_DEFAULT_START = 0

