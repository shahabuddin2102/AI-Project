from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, Request

from mcp.server.fastmcp import FastMCP
from mcp.types import TextContent

from dotenv import load_dotenv

import pymysql
import json
import httpx
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGSEARCH_API_KEY = os.getenv("LANGSEARCH_API_KEY")
# LANGSEARCH_API_KEY = os.getenv("GROQ_KEY")
LANGSEARCH_ENDPOINT = "https://api.langsearch.com/v1/web-search"

app = FastAPI(title="Groq SQL MCP Server with MySQL")
mcp = FastMCP(name="GroqSQLTools")

if not LANGSEARCH_API_KEY:
    raise ValueError("Lansearch api key is missing")

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
async def groq_call_sql(prompt: str) -> str:
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
# @mcp.tool()
# async def ask_from_database(query: str) -> dict:
#     """Converts user query to SQL using Groq, runs it on MySQL, and returns results."""
#     try:
#         sql = await groq_call_sql(query)

#         with db_connection.cursor() as cursor:
#             cursor.execute(sql)
#             rows = cursor.fetchall()

#         return {
#             "query": sql,
#             "result": rows or "No matching records found."
#         }

#     except Exception as e:
#         return {"error": str(e)}

@mcp.tool()
async def ask_from_database(query: str) -> list:
    """Run SQL query generated from natural language and return results with data_source tag"""
    try:
        sql = await groq_call_sql(query)

        with db_connection.cursor() as cursor:
            cursor.execute(sql)
            rows = cursor.fetchall()

        if not rows:
            return []

        # Add data_source to each row
        return [{"data_source": "database", **row} for row in rows]

    except Exception as e:
        return [{"type": "error", "text": str(e)}]
    

@mcp.tool()
async def ask_from_website(query: str) -> dict:
    """Search the web using LangSearch API and return concise result + source link."""
    langsearch_api_key = os.getenv("LANGSEARCH_API_KEY")

    try:
        headers = {
            "Authorization": f"Bearer {langsearch_api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "query": query,
            "freshness": "noLimit",
            "summary": True,
            "count": 1
        }

        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.post(
                "https://api.langsearch.com/v1/web-search",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            # print(result, "===============================")

            # Get the first result dict safely
            search_results = result.get("data", {}).get("webPages", {}).get("value", [])

            output = []
            for item in search_results[:3]:
                full_summary = item.get("summary", "No summary found.")
                source_url = item.get("url", "No source URL available.")
                concise_summary = ". ".join(full_summary.split(". ")[:3])

                output.append({
                    "data_source": "web_result",
                    "summary": concise_summary.strip(),
                    "source": source_url
                })
            # print(output, "-----------------------------------")

            return output

    except Exception as e:
        return [{
            "type": "error",
            "summary": f"Error: {str(e)}",
            "source": "N/A"
        }]

# === FastAPI App Mount ===

async def call_groq_llm(prompt: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {
                "role": "system",
                "content": "You're a smart AI agent. Decide which tool to call first based on user query. Use 'ask_from_database' for structured internal data, and fallback to 'ask_from_website' if no data is found."

            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 512,
        "top_p": 1,
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            response = await client.post(url, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            print(result, "------------------------------")
            return result["choices"][0]["message"]["content"]
    except httpx.ConnectTimeout:
        return "Error: Groq API timeout. Please try again later."
    
def is_empty_result(result):
    if not result:
        return True

    # 🔍 Step 1: Handle TextContent (single)
    if isinstance(result, TextContent):
        try:
            result = json.loads(result.text)
        except:
            return True

    # 🔍 Step 2: Handle list of TextContent
    if isinstance(result, list):
        parsed_list = []
        for item in result:
            if isinstance(item, TextContent):
                try:
                    parsed_item = json.loads(item.text)
                    parsed_list.append(parsed_item)
                except:
                    continue
            else:
                parsed_list.append(item)

        result = parsed_list

        return all(
            isinstance(r, dict) and all(
                v in [None, "", [], {}, "No matching records found."]
                for v in r.values()
            )
            for r in result
        )

    # 🔍 Step 3: Handle dict
    if isinstance(result, dict):
        return all(
            v in [None, "", [], {}, "No matching records found."]
            for v in result.values()
        )

    return False


@app.post("/mcp/auto-answer")
async def auto_answer(request: Request):
    body = await request.json()
    query = body.get("query", "")

    # === Tool Selection Prompt ===
    tool_selection_prompt = f"""
You're an intelligent AI router.

Available tools:
- ask_from_database → for structured MySQL database questions.
- ask_from_website → for web-based queries (via LangSearch API).

Instructions:
Return valid JSON in this format:
{{
  "tool": "ask_from_database",
  "arguments": {{
    "query": "{query}"
  }}
}}

Respond only with JSON.

User query: {query}
"""

    try:
        # Step 1: Call Groq LLM to choose tool
        llm_response = await call_groq_llm(tool_selection_prompt)
        print(f"llm_response is: {llm_response}")
        parsed = json.loads(llm_response)
        tool_name = parsed.get("tool")
        arguments = parsed.get("arguments", {})

        if not tool_name:
            raise ValueError("Tool name missing from LLM response.")

        # Step 2: First try database
        db_result = await mcp.call_tool(name="ask_from_database", arguments=arguments)

        print("DB RESULT:", db_result)

        # Step 3: If DB result is empty, fallback to website
        if is_empty_result(db_result):
            web_result = await mcp.call_tool(name="ask_from_website", arguments={"query": query})
            print("WEB RESULT:", web_result)

            if is_empty_result(web_result):
                final_result = [{"type": "text", "text": "No data found in both database and website tools."}]
            else:
                final_result = web_result
        else:
            final_result = db_result

    except Exception as e:
        final_result = [{"type": "error", "text": f"Tool selection or execution failed:\n{str(e)}"}]

    # Serialize response
    def serialize(obj):
        if isinstance(obj, TextContent):
            try:
                return json.loads(obj.text)
            except:
                return {"text": obj.text}
        return obj

    return JSONResponse(content=json.loads(json.dumps(final_result, default=serialize)))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run (app, host="127.0.0.1", port=8000)

