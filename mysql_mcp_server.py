from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from contextlib import contextmanager
import pymysql
import httpx
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(title="Groq SQL MCP Server with MySQL")
mcp = FastMCP(name="GroqSQLTools")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_connection
    db_connection = pymysql.connect(
        host="localhost",
        user="root",
        password="root",
        database="latest_database",
        cursorclass=pymysql.cursors.DictCursor,
        autocommit=True
    )
    yield
    db_connection.close()

app = FastAPI(title="Groq SQL MCP Server with MySQL", lifespan=lifespan)


# === Groq LLM for Natural Language to SQL ===
async def convert_prompt_to_sql(prompt: str) -> str:
    system_prompt = """You are an AI SQL assistant. Convert the user's natural language request into a SQL query.
Only use the table `test` with columns `id`, `question`, `answer`. Do not add explanations."""

    payload = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "model": "llama3-8b-8192"
    }

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient() as client:
        response = await client.post("https://api.groq.com/openai/v1/chat/completions", json=payload, headers=headers)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()

# === MCP Tool: Ask Query via LLM ===
@mcp.tool()
async def ask_faq_query(query: str) -> dict:
    """Converts user query to SQL using Groq, runs it on MySQL, and returns results."""
    try:
        sql = await convert_prompt_to_sql(query)

        with db_connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

        return {
            "query": sql,
            "result": rows or "No matching records found."
        }

    except Exception as e:
        return {"error": str(e)}
    
print("MCP Tool mounted on /mcp")
print("Registered tools:", mcp.tool)

# === Mount MCP Server ===
app.mount("/mcp", mcp.streamable_http_app())


@app.post("/ask-faq")
async def ask_faq_endpoint(data: dict):
    query = data.get("query")
    return await ask_faq_query(query)

# === Run Uvicorn ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
